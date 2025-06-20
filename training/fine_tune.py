import torch
import pymongo
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from apscheduler.schedulers.background import BackgroundScheduler

# --- MongoDB Setup ---
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["COD1"]
captures_col = db["captures"]

# --- Constants ---
EMBEDDING_SIZE = 512
MIN_IMAGES_PER_STUDENT = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# --- Dataset Definition ---
class EmbeddingDataset(Dataset):
    def __init__(self, records):
        self.embeddings = np.array([rec['embedding'] for rec in records], dtype=np.float32)
        self.labels, self.class_to_idx = self._encode_labels([rec['student_id'] for rec in records])

    def _encode_labels(self, label_list):
        unique = sorted(set(label_list))
        mapping = {label: idx for idx, label in enumerate(unique)}
        encoded = np.array([mapping[label] for label in label_list], dtype=np.int64)
        return encoded, mapping

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# --- Classification Head ---
class ClassificationHead(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)

# --- Average Embedding Update ---
def update_average_embeddings():
    print("Updating average embeddings...")
    student_embeddings = {}
    for rec in captures_col.find():
        sid = rec['student_id']
        emb = np.array(rec['embedding'], dtype=np.float32)
        student_embeddings.setdefault(sid, []).append(emb)

    for sid, embs in student_embeddings.items():
        if len(embs) >= MIN_IMAGES_PER_STUDENT:
            avg_emb = np.mean(embs, axis=0)
            db.embeddings.update_one(
                {"student_id": sid},
                {"$set": {"embedding": avg_emb.tolist()}},
                upsert=True
            )

# --- Training Logic ---
def train_classifier():
    print(f"[INFO] Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    records = list(captures_col.find())
    student_counts = {}
    for r in records:
        sid = r['student_id']
        student_counts[sid] = student_counts.get(sid, 0) + 1
    
    filtered = [r for r in records if student_counts[r['student_id']] >= MIN_IMAGES_PER_STUDENT]
    if len(filtered) < 10:
        print("[WARNING] Not enough data to train. Skipping...")
        return

    dataset = EmbeddingDataset(filtered)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ClassificationHead(EMBEDDING_SIZE, len(dataset.class_to_idx))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': dataset.class_to_idx
    }, "classification_head.pth")
    print("Model saved as classification_head.pth")
    update_average_embeddings()

# --- Start APScheduler Job ---
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(train_classifier, 'interval', weeks=2)
    scheduler.start()
    print("[INFO] Biweekly fine-tuning scheduler started.")
    return scheduler
