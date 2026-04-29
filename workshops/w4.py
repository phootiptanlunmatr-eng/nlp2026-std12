import os
import csv
import random
import numpy as np
from collections import Counter

import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler

# =============================================================================
# 1. สร้างพจนานุกรมคำพ้องความหมาย (Pokemon Data Augmentation)
# =============================================================================
POKEMON_SYNONYMS = {
    "Caught": ["Captured", "Gotcha", "Success"],
    "Escaped": ["Fled", "Ran away", "Broke free"],
    "Legendary": ["Mythical", "Rare encounter"]
}

def augment_pokemon_text(text):
    words = text.split()
    new_words = words.copy()
    for i, word in enumerate(words):
        if word in POKEMON_SYNONYMS:
            new_words[i] = random.choice(POKEMON_SYNONYMS[word])
    return " ".join(new_words)

# =============================================================================
# 2. โหลดข้อมูลจริง & ทำ SMOTE with Fallback
# =============================================================================
def load_pokemon_data(file_path):
    X, y = [], []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {file_path}")

    # Map ข้อมูลตัวอักษรให้เป็นตัวเลข
    tier_map = {"Legendary": 1.0, "Rare": 0.5, "Common": 0.1}
    ball_map = {"Ultra Ball": 1.0, "Great Ball": 0.5, "Poke Ball": 0.2}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hp = float(row['HP_Percent']) / 100.0
            lvl = float(row['Level']) / 100.0
            tier = tier_map.get(row['Pokemon_Tier'], 0.1)
            ball = ball_map.get(row['Pokeball_Type'], 0.2)
            status = 0.0 if row['Status_Effect'] == 'None' else 1.0
            
            X.append([hp, lvl, tier, ball, status]) # 5 Features
            y.append(1 if row['Catch_Result'] == 'Caught' else 0)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def balance_pokemon_data(X, y):
    counts = Counter(y)
    min_samples = min(counts.values())
    
    if min_samples > 1:
        # ใช้ k_neighbors ไม่เกินจำนวนข้อมูลที่มีลบ 1
        k_neighbors = min(5, min_samples - 1)
        if k_neighbors < 1: k_neighbors = 1
        sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)
    else:
        sampler = RandomOverSampler(random_state=42)
        
    X_res, y_res = sampler.fit_resample(X, y)
    print(f"⚖️ ข้อมูลหลังทำ SMOTE: {Counter(y_res)}")
    return X_res, y_res

# โหลดข้อมูลและจัดการ Imbalance
X_raw, y_raw = load_pokemon_data('raw_data.csv')
X_res, y_res = balance_pokemon_data(X_raw, y_raw)

# แปลงข้อมูลเป็น Tensor 3 มิติ สำหรับ LSTM (Batch, Seq_len, Input_size)
X_res_tensor = torch.tensor(X_res, dtype=torch.float32)
X_res_3d = X_res_tensor.unsqueeze(1) # เพิ่มมิติ Seq_len = 1
y_res_tensor = torch.tensor(y_res, dtype=torch.long)

print(f"📦 Shape สำหรับนำเข้า LSTM (3D): {X_res_3d.shape}")

# =============================================================================
# 3. สร้าง Models: BiLSTM & LSTM
# =============================================================================
class PokemonBiLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=2):
        super(PokemonBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        nn.init.xavier_uniform_(self.fc.weight) # อัปเดตเพิ่ม _ ตาม Pytorch สมัยใหม่

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1) # Mean Pooling
        return self.fc(pooled)

class PokemonLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=2):
        super(PokemonLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)

# =============================================================================
# 4. ฟังก์ชันเทรนและประเมินผล (Training & Evaluation)
# =============================================================================
def train_and_evaluate(model_class, name, X, y, class_names):
    print(f"\n🚀 เริ่มเทรนโมเดล: {name} ....")
    model = model_class(input_dim=5, hidden_dim=32, output_dim=2)
    
    # Cost-Sensitive Weight (ถ้าทายว่าหนี แต่จับได้ ไม่แย่เท่า ทายว่าจับได้ แต่หนีไป)
    weights = torch.tensor([1.5, 1.0]) 
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training Loop (รัน 100 Epoch เพื่อให้เห็นผลชัดขึ้น)
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
    # Evaluate Phase
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_pred = torch.argmax(logits, dim=1).numpy()
    
    cm = confusion_matrix(y.numpy(), y_pred)

    # Plot Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Confusion Matrix: {name}")
    ax.set_ylabel("Actual Result")
    ax.set_xlabel("Predicted Result")
    plt.tight_layout()
    plt.show()
    
    report = classification_report(y.numpy(), y_pred, target_names=class_names)
    print(f"📊 Classification Report: {name}\n")
    print(report)

# =============================================================================
# 5. รันเปรียบเทียบ
# =============================================================================
if __name__ == "__main__":
    class_list = ['Escaped (0)', 'Caught (1)']
    
    # รัน LSTM แบบธรรมดา
    train_and_evaluate(PokemonLSTM, "Unidirectional LSTM", X_res_3d, y_res_tensor, class_list)
    
    # รัน BiLSTM (อ่านไปกลับ) เพื่อเปรียบเทียบ
    train_and_evaluate(PokemonBiLSTM, "Bidirectional LSTM", X_res_3d, y_res_tensor, class_list)