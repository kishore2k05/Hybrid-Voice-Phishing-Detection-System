import os
import sys
import numpy as np
import pandas as pd #type: ignore
from pathlib import Path
import torch #type: ignore
import joblib #type: ignore
import gc #type: ignore
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig #type: ignore
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

LABEL_NAMES = ['neutral', 'slightly_suspicious', 'scam']
BERT_MODEL  = 'bert-base-multilingual-cased'
RF_WEIGHT   = 0.40
BERT_WEIGHT = 0.60

class VishingTester:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

    def load_models(self):
        print("\n" + "="*40 + "\nLOADING MODELS\n" + "="*40)
        
        try:
            rf_path = os.path.join(self.model_dir, 'rf_stacked_model.pkl')
            tfidf_path = os.path.join(self.model_dir, 'tfidf_vectorizer_final.pkl')
            
            rf_bundle = joblib.load(rf_path)
            self.rf = rf_bundle['rf1']
            self.rf.n_jobs = 1
            self.tfidf = joblib.load(tfidf_path)
            print("RF & TF-IDF loaded.")
        except Exception as e:
            print(f"Error loading RF/TF-IDF: {e}")
            sys.exit(1)

        try:
            bert_path = os.path.join(self.model_dir, 'bert_finetuned.pth')
            state_dict = torch.load(bert_path, map_location='cpu')
            
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

            config = BertConfig.from_pretrained(BERT_MODEL, num_labels=3)
            self.bert_model = BertForSequenceClassification(config) 
            self.bert_model.load_state_dict(state_dict, strict=False)
            
            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
            self.bert_model.eval()
            print("BERT model initialized and weights mapped.")
        except Exception as e:
            print(f"Error loading BERT: {e}")
            sys.exit(1)

    def predict(self, text):
        X = self.tfidf.transform([text])
        rf_probs = self.rf.predict_proba(X)[0]

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            bert_probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

        combined = (RF_WEIGHT * rf_probs) + (BERT_WEIGHT * bert_probs)
        pred_idx = np.argmax(combined)
        del inputs
        gc.collect()

        return {
            'prediction': LABEL_NAMES[pred_idx],
            'confidence': float(combined[pred_idx]),
            'ensemble_probs': combined,
            'rf_probs': rf_probs,
            'bert_probs': bert_probs
        }

    def load_transcripts(self, path):
        t_dir = Path(path)
        transcripts = []
        if not t_dir.exists():
            print(f"Path not found: {path}")
            return transcripts
            
        for f in sorted(t_dir.glob('*.txt')):
            text = f.read_text(encoding='utf-8').strip()
            if text:
                fname = f.name.lower()
                label = 'scam' if ('scam' in fname or 'microsoft' in fname) else 'neutral'
                transcripts.append({'filename': f.name, 'text': text, 'true_label': label})
        print(f"Loaded {len(transcripts)} transcripts.")
        return transcripts

    def run_test(self, transcripts, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print("\n" + "="*40 + "\nTESTING TRANSCRIPTS\n" + "="*40)
        
        results = []
        correct_count = 0
        total = len(transcripts)

        for i, t in enumerate(transcripts, 1):
            print(f"[{i}/{total}] {t['filename']}")
            try:
                res = self.predict(t['text'])
                is_correct = (res['prediction'] == t['true_label'])
                if is_correct: correct_count += 1
                
                p_neutral = res['ensemble_probs'][0] * 100
                p_suspicious = res['ensemble_probs'][1] * 100
                p_scam = res['ensemble_probs'][2] * 100

                print(f"   - Neutral:            {p_neutral:>6.2f}%")
                print(f"   - Slightly Suspicious: {p_suspicious:>6.2f}%")
                print(f"   - Scam:               {p_scam:>6.2f}%")
                print(f"   RESULT: {res['prediction'].upper()} (True: {t['true_label']}) {'✅' if is_correct else '❌'}")
                print("-" * 40)

                results.append({
                    'filename': t['filename'],
                    'true_label': t['true_label'],
                    'predicted': res['prediction'],
                    'neutral_pct': p_neutral,
                    'suspicious_pct': p_suspicious,
                    'scam_pct': p_scam,
                    'correct': is_correct
                })
            except Exception as e:
                print(f"   FAILED: {e}")
        
        csv_path = os.path.join(out_dir, "results.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"Detailed CSV saved to: {csv_path}")

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'datasets', 'cleaned_transcripts'))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'test_results')
    tester = VishingTester(model_dir=MODELS_DIR)
    tester.load_models()
    
    files = tester.load_transcripts(DATA_DIR)
    if files:
        tester.run_test(files, OUTPUT_DIR)
    else:
        print("No transcripts found.")