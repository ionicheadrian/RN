import pickle
import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

#facem clasa Dataset
class ExtendedMNISTDataset(Dataset):
    def __init__(self, root: str = "../input", train: bool = True):
        file_name = "extended_mnist_test.pkl" if not train else "extended_mnist_train.pkl"
        file_path = os.path.join(root, file_name)
        
        with open(file_path, "rb") as fp:
            self.data = pickle.load(fp)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]

# Load data
print("Loading training data...")
train_data = []
train_labels = []
for image, label in ExtendedMNISTDataset(train=True):
    img = np.array(image, dtype=np.float32).flatten()
    train_data.append(img)
    train_labels.append(label)

print("Loading test data...")
test_data = []
for image, label in ExtendedMNISTDataset(train=False):
    img = np.array(image, dtype=np.float32).flatten()
    test_data.append(img)

train_data = np.array(train_data, dtype=np.float32)
train_labels = np.array(train_labels, dtype=np.int64)
test_data = np.array(test_data, dtype=np.float32)

print(f"Train: {train_data.shape}, Test: {test_data.shape}")

# 2. Preprocessing
train_data = train_data / 255.0
test_data = test_data / 255.0

mean = train_data.mean(axis=0, keepdims=True).astype(np.float32)
std = train_data.std(axis=0, keepdims=True).astype(np.float32) + 1e-8
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# Data Augmentation
print("Augmenting data...")
augmented_data = [train_data]
augmented_labels = [train_labels]

for noise_level in [0.05, 0.1]:
    noisy = train_data + np.random.normal(0, noise_level, train_data.shape).astype(np.float32)
    augmented_data.append(noisy)
    augmented_labels.append(train_labels)

train_data_aug = np.vstack(augmented_data).astype(np.float32)
train_labels_aug = np.concatenate(augmented_labels).astype(np.int64)

print(f"Augmented train: {train_data_aug.shape}")

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    train_data_aug,
    train_labels_aug,
    test_size=0.1,
    random_state=42,
    stratify=train_labels_aug
)

# Convert to PyTorch tensors - EXPLICIT float32
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).long()
X_val_t = torch.from_numpy(X_val).float()
y_val_t = torch.from_numpy(y_val).long()
X_test_t = torch.from_numpy(test_data).float()

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)

# 3. DataLoaders
batch_size = 256
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# 4. Improved MLP model
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc5(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ImprovedMLP(input_dim=train_data.shape[1]).to(device)

# 5. Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=50,
    steps_per_epoch=len(train_loader),
    pct_start=0.3
)

# 6. Training loop
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return loss_sum / total, correct / total

epochs = 50
best_val_acc = 0.0
best_state = None
patience = 10
no_improve = 0

print("\nTraining...")
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    total_train = 0
    correct_train = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct_train += (preds == yb).sum().item()
        total_train += xb.size(0)

    train_loss = running_loss / total_train
    train_acc = correct_train / total_train

    val_loss, val_acc = evaluate(val_loader)

    print(f"Epoch {epoch:02d}/{epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict().copy()
        no_improve = 0
    else:
        no_improve += 1
        
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)

# 7. Test Time Augmentation
print("TTA")
model.eval()
tta_predictions = []

with torch.no_grad():
    # Original
    test_loader = DataLoader(TensorDataset(X_test_t), batch_size=batch_size, shuffle=False)
    all_preds = []
    for (xb,) in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        all_preds.append(F.softmax(logits, dim=1).cpu().numpy())
    tta_predictions.append(np.vstack(all_preds))
    
    # With noise augmentation
    for i in range(4):
        X_test_noisy = X_test_t + torch.randn_like(X_test_t) * 0.05
        test_loader_noisy = DataLoader(TensorDataset(X_test_noisy), batch_size=batch_size, shuffle=False)
        all_preds = []
        for (xb,) in test_loader_noisy:
            xb = xb.to(device)
            logits = model(xb)
            all_preds.append(F.softmax(logits, dim=1).cpu().numpy())
        tta_predictions.append(np.vstack(all_preds))

# Average predictions
avg_proba = np.mean(tta_predictions, axis=0)
final_predictions = np.argmax(avg_proba, axis=1)

# Save submission
predictions_csv = {
    "ID": list(range(len(final_predictions))),
    "target": final_predictions.tolist(),
}
df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)

print("submission saved")