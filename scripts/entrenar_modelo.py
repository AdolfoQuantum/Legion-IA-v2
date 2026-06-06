import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

# ============================================================
# LEGION IA — Entrenamiento LSTM v4 (4 clases)
# VERDE / AZUL / AMARILLO / ROJO
# ============================================================

BASE    = Path(r"C:\Users\accas\legion-ia")
DATASET = BASE / "modelos" / "dataset"
MODELOS = BASE / "modelos"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASES = ["VERDE", "AZUL", "AMARILLO", "ROJO"]

print(f"🖥️  Dispositivo: {DEVICE}")

# 1. Cargar dataset v4
print("\n📂 Cargando dataset v4...")
X = np.load(DATASET / "X_skeleton_v4.npy")
y = np.load(DATASET / "y_skeleton_v4.npy")

print(f"  X: {X.shape}")
print(f"  Clase 0 VERDE:    {np.sum(y==0)}")
print(f"  Clase 1 AZUL:     {np.sum(y==1)}")
print(f"  Clase 2 AMARILLO: {np.sum(y==2)}")
print(f"  Clase 3 ROJO:     {np.sum(y==3)}")

# 2. Dividir dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\n📊 División:")
print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# 3. Tensores
def to_tensor(X, y):
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long))

train_loader = DataLoader(to_tensor(X_train, y_train),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(to_tensor(X_val,   y_val),   batch_size=32)
test_loader  = DataLoader(to_tensor(X_test,  y_test),  batch_size=32)

# 4. Modelo LSTM v4 (4 clases)
class LegionLSTM(nn.Module):
    def __init__(self, input_size=68, hidden_size=128,
                 num_layers=2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True, dropout=0.3)
        self.bn   = nn.BatchNorm1d(hidden_size)
        self.fc1  = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2  = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.relu(self.fc1(out))
        out = self.drop(out)
        return self.fc2(out)

model     = LegionLSTM().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5)

print(f"\n🧠 Modelo LSTM v4 — 4 clases (VERDE/AZUL/AMARILLO/ROJO)")
print(f"  Parámetros: {sum(p.numel() for p in model.parameters()):,}")

# 5. Entrenamiento
def evaluar(loader):
    model.eval()
    correctos, total, loss_total = 0, 0, 0.0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            out  = model(X_b)
            loss = criterion(out, y_b)
            loss_total += loss.item()
            correctos  += (out.argmax(1) == y_b).sum().item()
            total      += len(y_b)
    return loss_total / len(loader), correctos / total

print("\n🚀 Entrenando...")
mejor_val_acc = 0.0
paciencia     = 0
PACIENCIA_MAX = 10

for epoch in range(1, 51):
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()

    train_loss, train_acc = evaluar(train_loader)
    val_loss,   val_acc   = evaluar(val_loader)
    scheduler.step(val_loss)

    print(f"  Epoch {epoch:02d}/50 — "
          f"train_acc: {train_acc*100:.1f}% — "
          f"val_acc: {val_acc*100:.1f}%")

    if val_acc > mejor_val_acc:
        mejor_val_acc = val_acc
        torch.save(model.state_dict(),
                   MODELOS / "legion_lstm_v4_best.pth")
        paciencia = 0
    else:
        paciencia += 1
        if paciencia >= PACIENCIA_MAX:
            print(f"\n  Early stopping en epoch {epoch}")
            break

# 6. Evaluación final
print(f"\n📈 Mejor val_accuracy: {mejor_val_acc*100:.2f}%")
model.load_state_dict(torch.load(
    MODELOS / "legion_lstm_v4_best.pth",
    map_location=DEVICE))
_, test_acc = evaluar(test_loader)
print(f"  Test accuracy: {test_acc*100:.2f}%")

# 7. Reporte por clase
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for X_b, y_b in test_loader:
        preds = model(X_b.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_b.numpy())

print("\n📋 Reporte por clase:")
print(classification_report(all_labels, all_preds,
      target_names=CLASES))

# 8. Guardar modelo final
torch.save(model.state_dict(), MODELOS / "legion_lstm_v4_final.pth")
print("✅ Modelo v4 guardado en modelos/legion_lstm_v4_final.pth")