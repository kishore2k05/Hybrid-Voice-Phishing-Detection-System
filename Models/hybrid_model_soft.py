import os
import pandas as pd #type: ignore
import joblib #type: ignore
import numpy as np #type: ignore
import torch #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from transformers import BertTokenizer, BertForSequenceClassification #type: ignore
from torch.optim import AdamW #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer #type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore
from sklearn.linear_model import LogisticRegression #type: ignore
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score #type: ignore
from sklearn.calibration import CalibratedClassifierCV #type: ignore
from sklearn.utils.class_weight import compute_class_weight #type: ignore
from tqdm import tqdm #type: ignore

try:
    from sklearn.frozen import FrozenEstimator #type: ignore
except ImportError:
    FrozenEstimator = None

MODELS_DIR      = os.path.join(os.path.dirname(__file__), 'saved_models_soft')
os.makedirs(MODELS_DIR, exist_ok=True)

STAGE_1_DATA    = '../datasets/dataset_full_combined.csv'
STAGE_2_DATA    = '../datasets/dataset_fixed_v3.csv'
BERT_MODEL      = 'bert-base-multilingual-cased'
BATCH_SIZE      = 16
LR_PHASE1       = 2e-5
LR_PHASE2       = 5e-6
EPOCHS_PHASE1   = 4
EPOCHS_PHASE2   = 3
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
LABEL_NAMES     = ['neutral', 'slightly_suspicious', 'scam']

RF_MODEL_FILE   = os.path.join(MODELS_DIR, 'soft_rf_stacked_model.pkl')
VECTORIZER_FILE = os.path.join(MODELS_DIR, 'soft_tfidf_vectorizer_final.pkl')
BERT_FILE       = os.path.join(MODELS_DIR, 'soft_bert_finetuned.pth')
ENSEMBLE_FILE   = os.path.join(MODELS_DIR, 'soft_ensemble_weights.pkl')
PHASE1_CKPT     = os.path.join(MODELS_DIR, 'soft_phase1_best.pth')
PHASE2_CKPT     = os.path.join(MODELS_DIR, 'soft_phase2_best.pth')

MANUAL_WEIGHT_BOOST = {'slightly_suspicious': 2.5}

RF_WEIGHT   = 0.40
BERT_WEIGHT = 0.60

def compute_weights(y, boost_override=None):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))
    if boost_override:
        for cls, factor in boost_override.items():
            if cls in weight_dict:
                weight_dict[cls] *= factor
    return weight_dict

def print_metrics(y_true, y_pred):
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(classification_report(y_true, y_pred))

def train_rf_stack(df1, df2):
    print("── RF1 + RF2 + Stacker Training ──")

    df1_train, df1_meta = train_test_split(
        df1, test_size=0.20, stratify=df1['label_description'], random_state=42
    )
    df2_train, df2_temp = train_test_split(
        df2, test_size=0.40, stratify=df2['label_description'], random_state=42
    )
    df2_meta, df2_test = train_test_split(
        df2_temp, test_size=0.50, stratify=df2_temp['label_description'], random_state=42
    )

    vectorizer = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 3),
        sublinear_tf=True, min_df=2,
        strip_accents='unicode', analyzer='word'
    )
    vectorizer.fit(pd.concat([df1_train['dialogue'], df2_train['dialogue']]))

    X1_train = vectorizer.transform(df1_train['dialogue'])
    y1_train  = df1_train['label_description']
    rf1 = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=4,
        min_samples_split=10, max_features='sqrt',
        random_state=42, n_jobs=-1,
        class_weight=compute_weights(y1_train)
    )
    rf1.fit(X1_train, y1_train)
    print("RF1 validation:")
    X1_meta = vectorizer.transform(df1_meta['dialogue'])
    y1_meta  = df1_meta['label_description']
    print_metrics(y1_meta, rf1.predict(X1_meta))

    X2_train = vectorizer.transform(df2_train['dialogue'])
    y2_train  = df2_train['label_description']
    rf2 = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=4,
        min_samples_split=10, max_features='sqrt',
        random_state=42, n_jobs=-1,
        class_weight=compute_weights(y2_train, boost_override=MANUAL_WEIGHT_BOOST)
    )
    rf2.fit(X2_train, y2_train)
    print("RF2 validation:")
    X2_meta = vectorizer.transform(df2_meta['dialogue'])
    y2_meta  = df2_meta['label_description']
    print_metrics(y2_meta, rf2.predict(X2_meta))

    meta_X = np.hstack([rf1.predict_proba(X2_meta), rf2.predict_proba(X2_meta)])
    meta_y = y2_meta.values

    stacker = LogisticRegression(
        max_iter=1000, C=1.0, random_state=42,
        class_weight=compute_weights(meta_y, boost_override=MANUAL_WEIGHT_BOOST)
    )
    stacker.fit(meta_X, meta_y)

    if FrozenEstimator:
        calibrated_stacker = CalibratedClassifierCV(
            estimator=FrozenEstimator(stacker), method='sigmoid'
        )
    else:
        calibrated_stacker = CalibratedClassifierCV(
            stacker, cv='prefit', method='sigmoid'
        )
    calibrated_stacker.fit(meta_X, meta_y)

    joblib.dump({'rf1': rf1, 'rf2': rf2, 'stacker': calibrated_stacker}, RF_MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    return rf1, rf2, calibrated_stacker, vectorizer, df2_test

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

def make_loaders(df, tokenizer, label_col='label'):
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['dialogue'].tolist(), df[label_col].tolist(),
        test_size=0.15, stratify=df[label_col], random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )
    train_loader = DataLoader(ScamDataset(X_train, y_train, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ScamDataset(X_val, y_val, tokenizer), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(ScamDataset(X_test, y_test, tokenizer), batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, leave=False):
        optimizer.zero_grad()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_bert(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels.extend(batch['labels'].tolist())
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            preds.extend(torch.argmax(model(**batch).logits, dim=1).cpu().tolist())
    return preds, labels

def bert_predict_proba(model, texts, tokenizer):
    model.eval()
    dataset  = ScamDataset(texts, [0] * len(texts), tokenizer)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE)
    all_proba = []
    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            logits = model(**batch).logits
            proba  = torch.softmax(logits, dim=1).cpu().numpy()
            all_proba.append(proba)
    return np.vstack(all_proba)

def run_bert_phase(model, train_loader, val_loader, optimizer, epochs, checkpoint):
    best_acc = 0.0
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer)
        preds, labels = evaluate_bert(model, val_loader)
        val_acc = accuracy_score(labels, preds)
        print(f"  Epoch {epoch+1} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
    return best_acc

def train_bert(df1, df2):
    print("\n── BERT Phase 1 Training (Dataset 1) ──")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    train1, val1, test1 = make_loaders(df1, tokenizer)
    bert = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=3).to(DEVICE)

    opt1 = AdamW(bert.parameters(), lr=LR_PHASE1)
    run_bert_phase(bert, train1, val1, opt1, EPOCHS_PHASE1, PHASE1_CKPT)
    bert.load_state_dict(torch.load(PHASE1_CKPT))
    p1_preds, p1_labels = evaluate_bert(bert, test1)
    print(f"BERT Phase 1 Test Accuracy: {accuracy_score(p1_labels, p1_preds):.4f}")
    print(classification_report(p1_labels, p1_preds, target_names=LABEL_NAMES))

    print("\n── BERT Phase 2 Fine-Tuning (Dataset 2) ──")
    train2, val2, test2 = make_loaders(df2, tokenizer)
    opt2 = AdamW(bert.parameters(), lr=LR_PHASE2)
    run_bert_phase(bert, train2, val2, opt2, EPOCHS_PHASE2, PHASE2_CKPT)
    bert.load_state_dict(torch.load(PHASE2_CKPT))
    p2_preds, p2_labels = evaluate_bert(bert, test2)
    print(f"BERT Phase 2 Test Accuracy: {accuracy_score(p2_labels, p2_preds):.4f}")
    print(classification_report(p2_labels, p2_preds, target_names=LABEL_NAMES))
    torch.save(bert.state_dict(), BERT_FILE)
    return bert, tokenizer

def soft_vote(rf_proba, bert_proba, rf_weight=RF_WEIGHT, bert_weight=BERT_WEIGHT):
    combined = rf_weight * rf_proba + bert_weight * bert_proba
    return combined

def get_rf_proba(rf1, rf2, stacker, vectorizer, texts):
    X      = vectorizer.transform(texts)
    meta_X = np.hstack([rf1.predict_proba(X), rf2.predict_proba(X)])
    return stacker.predict_proba(meta_X)

def evaluate_ensemble(rf1, rf2, stacker, vectorizer, bert, tokenizer, df_test, label_col='label_description'):
    print("\n" + "="*60)
    print("  ⚖️  SOFT VOTING APPROACH")
    print(f"  RF Weight: {RF_WEIGHT*100:.0f}%  |  BERT Weight: {BERT_WEIGHT*100:.0f}%")
    print("  Predictions are based on weighted average of RF and")
    print("  BERT probabilities — NO learned meta-model is used.")
    print("="*60 + "\n")

    texts  = df_test['dialogue'].tolist()
    y_true = df_test[label_col].values

    rf_proba   = get_rf_proba(rf1, rf2, stacker, vectorizer, texts)
    bert_proba = bert_predict_proba(bert, texts, tokenizer)

    stacker_classes = stacker.classes_
    bert_col_map    = {name: i for i, name in enumerate(LABEL_NAMES)}
    bert_proba_aligned = np.column_stack([
        bert_proba[:, bert_col_map[cls]] for cls in stacker_classes
    ])

    combined = soft_vote(rf_proba, bert_proba_aligned)
    y_pred   = stacker_classes[np.argmax(combined, axis=1)]

    print("\n── Soft Voting Ensemble (RF {:.0f}% + BERT {:.0f}%) ──".format(
        RF_WEIGHT * 100, BERT_WEIGHT * 100))
    print_metrics(y_true, y_pred)

    joblib.dump({
        'rf_weight': RF_WEIGHT,
        'bert_weight': BERT_WEIGHT,
        'stacker_classes': stacker_classes
    }, ENSEMBLE_FILE)

def main():
    print("="*60)
    print("  🚀 RF + BERT Soft Voting Training")
    print("  ⚖️  This pipeline uses the SOFT VOTING APPROACH")
    print(f"  RF Weight: {RF_WEIGHT*100:.0f}%  |  BERT Weight: {BERT_WEIGHT*100:.0f}%")
    print("="*60 + "\n")
    print(f"📁 Models will be saved to: {MODELS_DIR}\n")

    df1 = pd.read_csv(STAGE_1_DATA).dropna(subset=['dialogue'])
    df2 = pd.read_csv(STAGE_2_DATA).dropna(subset=['dialogue'])

    rf1, rf2, stacker, vectorizer, df2_test = train_rf_stack(df1, df2)
    bert, tokenizer                          = train_bert(df1, df2)
    evaluate_ensemble(rf1, rf2, stacker, vectorizer, bert, tokenizer, df2_test)

    print(f"\nRF stack   → {RF_MODEL_FILE}")
    print(f"Vectorizer → {VECTORIZER_FILE}")
    print(f"BERT       → {BERT_FILE}")
    print(f"Ensemble   → {ENSEMBLE_FILE}")


if __name__ == "__main__":
    main()