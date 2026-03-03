import os
import numpy as np
import joblib # type: ignore
import torch # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from transformers import BertTokenizer, BertForSequenceClassification # type: ignore

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_models_soft')
BERT_MODEL = 'bert-base-multilingual-cased'
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LABEL_NAMES = ['neutral', 'slightly_suspicious', 'scam']
RF_WEIGHT = 0.40
BERT_WEIGHT = 0.60

RF_MODEL_FILE = os.path.join(MODELS_DIR, 'soft_rf_stacked_model.pkl')
VECTORIZER_FILE = os.path.join(MODELS_DIR, 'soft_tfidf_vectorizer_final.pkl')
BERT_FILE = os.path.join(MODELS_DIR, 'soft_bert_finetuned.pth')


class ScamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


_rf_data = None
_bert_model = None
_tokenizer = None


def _load_rf():
    global _rf_data
    if _rf_data is None:
        data = joblib.load(RF_MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        _rf_data = {
            'rf1': data['rf1'],
            'rf2': data['rf2'],
            'stacker': data['stacker'],
            'vectorizer': vectorizer,
        }
    return _rf_data


def _load_bert():
    global _bert_model, _tokenizer
    if _bert_model is None:
        _tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        _bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=3)
        _bert_model.load_state_dict(torch.load(BERT_FILE, map_location=DEVICE))
        _bert_model.to(DEVICE)
        _bert_model.eval()
    return _bert_model, _tokenizer


def _get_rf_proba(texts):
    rf = _load_rf()
    X = rf['vectorizer'].transform(texts)
    meta_X = np.hstack([rf['rf1'].predict_proba(X), rf['rf2'].predict_proba(X)])
    return rf['stacker'].predict_proba(meta_X)


def _get_bert_proba(texts):
    model, tokenizer = _load_bert()
    dataset = ScamDataset(texts, [0] * len(texts), tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    all_proba = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            logits = model(**batch).logits
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            all_proba.append(proba)
    return np.vstack(all_proba)


def predict(text):
    texts = [text]
    rf_proba = _get_rf_proba(texts)
    bert_proba = _get_bert_proba(texts)
    combined = RF_WEIGHT * rf_proba + BERT_WEIGHT * bert_proba
    probs = combined[0]

    result = {
        'neutral': float(probs[0]),
        'slightly_suspicious': float(probs[1]),
        'scam': float(probs[2]),
        'label': LABEL_NAMES[np.argmax(probs)],
        'confidence': float(np.max(probs)),
    }
    return result