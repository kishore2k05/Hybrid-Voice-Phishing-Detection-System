import os
import io
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch # type: ignore
import joblib # type: ignore
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig # type: ignore

LABEL_NAMES = ["neutral", "slightly_suspicious", "scam"]
BERT_MODEL = "bert-base-multilingual-cased"
RF_WEIGHT = 0.40
BERT_WEIGHT = 0.60

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models_soft")
RF_MODEL_FILE = os.path.join(MODELS_DIR, "soft_rf_stacked_model.pkl")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "soft_tfidf_vectorizer_final.pkl")
BERT_FILE = os.path.join(MODELS_DIR, "soft_bert_finetuned.pth")

_loaded = False
_rf1 = None
_rf2 = None
_stacker = None
_tfidf = None
_bert_model = None
_tokenizer = None


def _load_all():
    global _loaded, _rf1, _rf2, _stacker, _tfidf, _bert_model, _tokenizer

    if _loaded:
        return

    rf_bundle = joblib.load(RF_MODEL_FILE)
    _rf1 = rf_bundle["rf1"]
    _rf2 = rf_bundle["rf2"]
    _stacker = rf_bundle["stacker"]
    _rf1.n_jobs = 1
    _rf2.n_jobs = 1

    _tfidf = joblib.load(VECTORIZER_FILE)

    bert_path = BERT_FILE
    with open(bert_path, "rb") as f:
        state_dict = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=3)
    _bert_model = BertForSequenceClassification(config)
    _bert_model.load_state_dict(state_dict, strict=False)

    _tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    _bert_model.eval()

    _loaded = True


def predict(text):
    _load_all()

    X = _tfidf.transform([text])
    meta_X = np.hstack([_rf1.predict_proba(X), _rf2.predict_proba(X)])
    rf_probs = _stacker.predict_proba(meta_X)[0]

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = _bert_model(**inputs)
        bert_probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

    stacker_classes = list(_stacker.classes_)
    bert_col_map = {name: i for i, name in enumerate(LABEL_NAMES)}
    bert_aligned = np.array([bert_probs[bert_col_map[cls]] for cls in stacker_classes])

    combined = RF_WEIGHT * rf_probs + BERT_WEIGHT * bert_aligned
    pred_idx = np.argmax(combined)
    pred_label = stacker_classes[pred_idx]

    scores = {cls: float(combined[i]) for i, cls in enumerate(stacker_classes)}

    del inputs
    gc.collect()

    return {
        "neutral": scores.get("neutral", 0.0),
        "slightly_suspicious": scores.get("slightly_suspicious", 0.0),
        "scam": scores.get("scam", 0.0),
        "label": pred_label,
        "confidence": float(np.max(combined)),
    }