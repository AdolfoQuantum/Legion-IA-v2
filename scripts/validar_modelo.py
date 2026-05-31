import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, 
                             confusion_matrix, 
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# LEGION IA — Validación del modelo LSTM
# ============================================================

BASE    = Path(r"C:\Users\accas\legion-ia")
DATASET = BASE / "modelos" / "dataset"
MODELOS = BASE / "modelos"
RESULTS = BASE / "resultados"
RESULTS.mkdir(exist_ok=True)

CLASES = ["SEGURA", "PRECAUCION", "PELIGRO"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mismo modelo que en entrenar_modelo.py
class LegionLSTM(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
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

# 1. Cargar datos y modelo
print("📂 Cargando dataset y modelo...")
X = np.load(DATASET / "X_skeleton.npy")
y = np.load(DATASET / "y_skeleton.npy")

_, X_temp, _, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
_, X_test, _, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

model = LegionLSTM().to(DEVICE)
model.load_state_dict(torch.load(
    MODELOS / "legion_lstm_best.pth", 
    map_location=DEVICE))
model.eval()
print("  ✔ Modelo cargado")

# 2. Predicciones
test_ds     = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_ds, batch_size=32)

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        out   = model(X_b.to(DEVICE))
        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(y_b.numpy())

all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

# 3. Métricas
print("\n📋 REPORTE COMPLETO:")
print("="*55)
print(classification_report(all_labels, all_preds, 
                             target_names=CLASES, digits=3))

# 4. Análisis de confianza por clase
print("🎯 CONFIANZA PROMEDIO POR CLASE:")
print("="*55)
for i, clase in enumerate(CLASES):
    mask      = all_labels == i
    correctos = all_preds[mask] == i
    conf_ok   = all_probs[mask][correctos, i].mean() if correctos.sum() > 0 else 0
    conf_fail = all_probs[mask][~correctos, i].mean() if (~correctos).sum() > 0 else 0
    print(f"  {clase}:")
    print(f"    Predicciones correctas:   {conf_ok*100:.1f}% confianza promedio")
    print(f"    Predicciones incorrectas: {conf_fail*100:.1f}% confianza promedio")

# 5. Matriz de confusión
print("\n📊 Generando matriz de confusión...")
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASES)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title("Legion IA — Matriz de Confusión", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=150)
print(f"  ✔ Guardada en resultados/confusion_matrix.png")

# 6. Falsos negativos críticos (PELIGRO clasificado como SEGURA)
fn_criticos = np.sum((all_labels == 2) & (all_preds == 0))
total_peligro = np.sum(all_labels == 2)
print(f"\n⚠️  ANÁLISIS DE RIESGO:")
print(f"  Falsos negativos críticos")
print(f"  (PELIGRO → SEGURA): {fn_criticos}/{total_peligro} "
      f"({fn_criticos/total_peligro*100:.1f}%)")

plt.show()
print("\n✅ Validación completa")
