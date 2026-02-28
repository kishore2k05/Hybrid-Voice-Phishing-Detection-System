import pandas as pd #type: ignore
import joblib #type: ignore
import numpy as np #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer #type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore
from sklearn.metrics import classification_report, balanced_accuracy_score #type: ignore
from sklearn.calibration import CalibratedClassifierCV #type: ignore
from sklearn.utils.class_weight import compute_class_weight#type: ignore

try:
    from sklearn.frozen import FrozenEstimator #type: ignore
except ImportError:
    FrozenEstimator = None

STAGE_1_DATA    = '../datasets/dataset_full_combined.csv'
STAGE_2_DATA    = '../datasets/dataset_fixed_v3.csv'
MODEL_FILE      = 'vishing_model_final.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer_final.pkl'

MANUAL_WEIGHT_BOOST = {
    'slightly_suspicious': 2.5
}


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
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy: {bal_acc * 100:.2f}%")
    print(classification_report(y_true, y_pred))


def find_best_thresholds(model, X, y):
    from sklearn.metrics import f1_score
    proba = model.predict_proba(X)
    classes = model.classes_
    best_thresholds = {}
    for i, cls in enumerate(classes):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.20, 0.81, 0.05):
            preds = np.where(proba[:, i] >= t, cls, '__other__')
            binary_true = np.where(y == cls, cls, '__other__')
            f1 = f1_score(binary_true, preds, pos_label=cls, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[cls] = best_t
    return best_thresholds


def predict_with_thresholds(model, X, thresholds):
    proba = model.predict_proba(X)
    classes = model.classes_
    adjusted = proba / np.array([thresholds[c] for c in classes])
    return classes[np.argmax(adjusted, axis=1)]


def train_two_stage_brain():
    print("🚀 Starting Two-Stage Training\n")

    df1 = pd.read_csv(STAGE_1_DATA).dropna(subset=['dialogue'])
    df2 = pd.read_csv(STAGE_2_DATA).dropna(subset=['dialogue'])

    df2_train, df2_temp = train_test_split(
        df2, test_size=0.35, stratify=df2['label_description'], random_state=42
    )
    df2_calib, df2_test = train_test_split(
        df2_temp, test_size=0.57, stratify=df2_temp['label_description'], random_state=42
    )

    df1_train, df1_val = train_test_split(
        df1, test_size=0.15, stratify=df1['label_description'], random_state=42
    )

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
        strip_accents='unicode',
        analyzer='word'
    )
    vectorizer.fit(pd.concat([df1_train['dialogue'], df2_train['dialogue']]))

    print("── Stage 1 Training ──")
    X1_train = vectorizer.transform(df1_train['dialogue'])
    y1_train  = df1_train['label_description']

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        min_samples_split=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        warm_start=True,
        class_weight=compute_weights(y1_train)
    )
    model.fit(X1_train, y1_train)

    X1_val  = vectorizer.transform(df1_val['dialogue'])
    y1_val  = df1_val['label_description']
    print_metrics(y1_val, model.predict(X1_val))

    print("── Stage 2 Fine-Tuning (warm start) ──")
    X2_train = vectorizer.transform(df2_train['dialogue'])
    y2_train  = df2_train['label_description']

    model.set_params(
        class_weight=compute_weights(y2_train, boost_override=MANUAL_WEIGHT_BOOST),
        n_estimators=model.n_estimators + 100
    )
    model.fit(X2_train, y2_train)

    X2_calib = vectorizer.transform(df2_calib['dialogue'])
    y2_calib  = df2_calib['label_description']

    if FrozenEstimator:
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(model), method='sigmoid'
        )
    else:
        calibrated_model = CalibratedClassifierCV(
            model, cv='prefit', method='sigmoid'
        )
    calibrated_model.fit(X2_calib, y2_calib)

    thresholds = find_best_thresholds(calibrated_model, X2_calib, y2_calib)

    X_test = vectorizer.transform(df2_test['dialogue'])
    y_test  = df2_test['label_description']

    y_pred_default = calibrated_model.predict(X_test)
    y_pred_tuned   = predict_with_thresholds(calibrated_model, X_test, thresholds)

    bal_default = balanced_accuracy_score(y_test, y_pred_default)
    bal_tuned   = balanced_accuracy_score(y_test, y_pred_tuned)

    print("Default thresholds (0.5):")
    print_metrics(y_test, y_pred_default)

    print("Tuned thresholds:")
    print_metrics(y_test, y_pred_tuned)

    use_tuned = bal_tuned > bal_default

    joblib.dump({
        'model': calibrated_model,
        'thresholds': thresholds if use_tuned else None,
        'use_tuned_thresholds': use_tuned
    }, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print(f"✅ Model saved  → {MODEL_FILE}")
    print(f"✅ Vectorizer saved → {VECTORIZER_FILE}")


if __name__ == "__main__":
    train_two_stage_brain()