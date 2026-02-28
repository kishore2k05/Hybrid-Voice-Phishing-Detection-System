import pandas as pd #type: ignore
import torch #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from transformers import BertTokenizer, BertForSequenceClassification #type: ignore
from torch.optim import AdamW #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import accuracy_score, classification_report #type: ignore
from tqdm import tqdm #type: ignore

DATASET_1 = '../datasets/dataset_fixed_v3.csv'
DATASET_2 = '../datasets/dataset_full_combined.csv'

MODEL         = 'bert-base-multilingual-cased'
BATCH_SIZE    = 16
LR_PRETRAIN   = 2e-5  
LR_FINETUNE   = 5e-6  
EPOCHS_PHASE1 = 4
EPOCHS_PHASE2 = 3
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def make_loaders(df, tokenizer):
    """Split a dataframe into train/val/test DataLoaders."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['dialogue'].tolist(),
        df['label'].tolist(),
        test_size=0.15,
        stratify=df['label'],
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.176,
        stratify=y_temp,
        random_state=42
    )
    train_loader = DataLoader(ScamDataset(X_train, y_train, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ScamDataset(X_val,   y_val,   tokenizer),
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(ScamDataset(X_test,  y_test,  tokenizer),
                              batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        optimizer.zero_grad()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            labels.extend(batch['labels'].tolist())
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
            outputs = model(**batch)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())
    return preds, labels


def run_training(model, train_loader, val_loader, optimizer, epochs, checkpoint_path):
    """Generic training loop that saves the best checkpoint."""
    best_acc = 0.0
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer)
        val_preds, val_labels = evaluate(model, val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"  Epoch {epoch+1} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best checkpoint (val_acc={val_acc:.4f})")
    return best_acc

tokenizer = BertTokenizer.from_pretrained(MODEL)

print("=" * 60)
print("PHASE 1 - Training on Dataset 1")
print("=" * 60)

df1 = pd.read_csv(DATASET_1)
train_loader1, val_loader1, test_loader1 = make_loaders(df1, tokenizer)

model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=3).to(DEVICE)
optimizer1 = AdamW(model.parameters(), lr=LR_PRETRAIN)

run_training(model, train_loader1, val_loader1,
             optimizer1, EPOCHS_PHASE1, 'phase1_best.pth')

model.load_state_dict(torch.load('phase1_best.pth'))

test_preds1, test_labels1 = evaluate(model, test_loader1)
print(f"\nPhase 1 Test Accuracy: {accuracy_score(test_labels1, test_preds1):.4f}")
print(classification_report(test_labels1, test_preds1,
      target_names=['neutral', 'slightly_suspicious', 'scam']))

print("=" * 60)
print("PHASE 2 - Fine-tuning on Dataset 2")
print("=" * 60)

df2 = pd.read_csv(DATASET_2)
train_loader2, val_loader2, test_loader2 = make_loaders(df2, tokenizer)

optimizer2 = AdamW(model.parameters(), lr=LR_FINETUNE)

run_training(model, train_loader2, val_loader2,
             optimizer2, EPOCHS_PHASE2, 'phase2_best.pth')

model.load_state_dict(torch.load('phase2_best.pth'))

test_preds2, test_labels2 = evaluate(model, test_loader2)
print(f"\nPhase 2 Test Accuracy: {accuracy_score(test_labels2, test_preds2):.4f}")
print(classification_report(test_labels2, test_preds2,
      target_names=['neutral', 'slightly_suspicious', 'scam']))