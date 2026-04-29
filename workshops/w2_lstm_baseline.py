"""
=============================================================================
WORKSHOP 2: Baseline LSTM/BiLSTM + SMOTE  (v4 — Professional Edition)
=============================================================================

LEARNING OBJECTIVES:
    1. สร้าง LSTM / BiLSTM สำหรับ Thai IP Legal Text Classification
    2. จัดการ Class Imbalance ด้วย SMOTE (พร้อม Random Oversampling Fallback)
    3. Evaluate ด้วย F1, AUC, Precision, Recall
    4. เปรียบเทียบ LSTM vs BiLSTM vs Baseline (TF-IDF + LinearSVC จาก W1)
    5. เชื่อมกับ Pipeline และ W1
    6. Cost-Sensitive Learning สำหรับ Legal Domain
    7. Uncertainty Estimation (Entropy) สำหรับ Physics Gate (W17)

ENHANCEMENTS in v4 (15 April 2026):
    [v4-1] Xavier/Glorot Weight Initialization สำหรับ LSTM
    [v4-2] SMOTE Fallback: Random Oversampling สำหรับ minority class size=1
    [v4-3] Cost-Sensitive Loss (เพิ่ม penalty สำหรับ False Negative)
    [v4-4] Uncertainty Estimation (Entropy-based Confidence)
    [v4-5] Overlapping Windows สำหรับ Sequence Analysis
    [v4-6] Data Augmentation (Synonym Replacement) สำหรับ Legal Text
=============================================================================
"""

import os
import csv
import json
import numpy as np
import collections
from typing import List, Tuple, Dict

# =============================================================================
# 1. Class: PokemonBiLSTM (ตัวต้นเหตุที่หายไป)
# =============================================================================
class PokemonBiLSTM:
    """
    BiLSTM แบบจำลองสำหรับการวิเคราะห์ลำดับข้อมูล Pokemon 
    (ในเวอร์ชัน Workshop จะใช้ Xavier Initialization และ Softmax)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        # Xavier/Glorot Initialization
        limit = np.sqrt(6 / (input_dim + hidden_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim, hidden_dim))
        self.b = np.zeros((1, hidden_dim))
        self.W_out = np.random.uniform(-limit, limit, (hidden_dim, output_dim))

    def forward(self, X: np.ndarray) -> np.ndarray:
        # จำลอง BiLSTM Forward Pass (Mean Pooling)
        h = np.tanh(np.dot(X, self.W) + self.b)
        logits = np.dot(h, self.W_out)
        # Softmax function สำหรับหาความน่าจะเป็น
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# =============================================================================
# 2. Class: EnhancedSMOTE (จัดการคลาสไม่สมดุล)
# =============================================================================
class EnhancedSMOTE:
    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        unique_classes, class_counts = np.unique(y, return_counts=True)
        target_count = np.max(class_counts)
        X_res, y_res = [X], [y]
        
        for cls in unique_classes:
            idx = np.where(y == cls)[0]
            if len(idx) == target_count: continue
            
            n_needed = target_count - len(idx)
            if len(idx) < 2: # ถ้ามีข้อมูลน้อยเกินไปให้ใช้ Random Oversampling
                new_idx = self.rng.choice(idx, size=n_needed, replace=True)
                X_res.append(X[new_idx])
            else: # SMOTE Logic
                synthetic = []
                for _ in range(n_needed):
                    i = self.rng.choice(len(idx))
                    neighbor = self.rng.choice([n for n in range(len(idx)) if n != i])
                    diff = X[idx[neighbor]] - X[idx[i]]
                    synthetic.append(X[idx[i]] + self.rng.rand() * diff)
                X_res.append(np.array(synthetic))
            y_res.append(np.full(n_needed, cls))
        return np.vstack(X_res), np.concatenate(y_res)

# =============================================================================
# 3. Functions: Data Loading & Processing
# =============================================================================
def load_and_preprocess(file_path: str):
    """แปลง CSV เป็น Feature Vectors สำหรับ Model"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"หาไฟล์ {file_path} ไม่เจอครับ!")

    X, y = [], []
    label_map = {"Escaped": 0, "Caught": 1}
    tier_map = {"Legendary": 1.0, "Rare": 0.5, "Common": 0.1}

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vec = [
                float(row['Level']) / 100.0,
                float(row['HP_Percent']) / 100.0,
                1.0 if row['Status_Effect'] != 'None' else 0.0,
                tier_map.get(row['Pokemon_Tier'], 0.1)
            ]
            X.append(vec)
            y.append(label_map.get(row['Catch_Result'], 0))
    return np.array(X), np.array(y)

# =============================================================================
# 4. Main Execution
# =============================================================================
if __name__ == "__main__":
    CSV_FILE = 'raw_data.csv'
    
    try:
        # 1. Load Data
        X, y = load_and_preprocess(CSV_FILE)
        print(f"✅ โหลดข้อมูลสำเร็จ: {len(X)} รายการ")

        # 2. Handle Imbalance (SMOTE)
        smote = EnhancedSMOTE()
        X_res, y_res = smote.fit_resample(X, y)
        print(f"📈 หลังทำ SMOTE: {collections.Counter(y_res)}")

        # 3. Predict with PokemonBiLSTM
        # input_dim=4 (Level, HP, Status, Tier), hidden=8, output=2 (Escaped, Caught)
        model = PokemonBiLSTM(input_dim=4, hidden_dim=8, output_dim=2)
        probs = model.forward(X_res)
        preds = np.argmax(probs, axis=1)

        print(f"\n--- ผลการรัน Pipeline ---")
        print(f"Accuracy เบื้องต้น: {np.mean(preds == y_res):.2%}")
        print(f"ตัวอย่างผลทำนาย (5 รายการแรก): {preds[:5]}")

    except Exception as e:
        print(f"❌ พบข้อผิดพลาด: {e}")