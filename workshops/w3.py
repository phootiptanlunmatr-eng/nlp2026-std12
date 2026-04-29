import numpy as np
import os
import csv

# --- 1. Utility Functions ---
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# --- 2. Transformer Components ---
class PokemonTransformerBlock:
    def __init__(self, seq_len, d_model):
        self.d_model = d_model
        self.seq_len = seq_len
        np.random.seed(42)
        
        self.pe = self._build_pe()
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        
        self.W_ff = np.random.randn(d_model, d_model) * 0.1
        self.W_out = np.random.randn(d_model, 2) * 0.1

    def _build_pe(self):
        pe = np.zeros((self.seq_len, self.d_model))
        for pos in range(self.seq_len):
            for i in range(0, self.d_model, 2):
                div = np.exp(i * -np.log(10000.0) / self.d_model)
                pe[pos, i] = np.sin(pos * div)
                if i+1 < self.d_model: pe[pos, i+1] = np.cos(pos * div)
        return pe

    def forward(self, x):
        x = x + self.pe 
        
        Q, K, V = np.dot(x, self.W_q), np.dot(x, self.W_k), np.dot(x, self.W_v)
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model)
        weights = softmax(scores)
        attention_out = np.dot(weights, V)
        
        ff_out = np.maximum(0, np.dot(attention_out, self.W_ff)) 
        pooled = np.mean(ff_out, axis=0) 
        logits = np.dot(pooled, self.W_out)
        
        return softmax(logits.reshape(1, -1)), weights

# --- 3. Data Processing ---
def prepare_data_with_names(file_path, d_model=8):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {file_path}")

    tier_map = {"Legendary": 1.0, "Rare": 0.5, "Common": 0.1}
    ball_map = {"Ultra Ball": 1.0, "Great Ball": 0.5, "Poke Ball": 0.2}
    status_map = {"None": 0.0, "Sleep": 1.0, "Paralyze": 0.8, "Poison": 0.5}

    with open(file_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    unique_names = list(set([row['Pokemon_Name'] for row in rows]))
    name_to_id = {name: idx for idx, name in enumerate(unique_names)}
    
    np.random.seed(42)
    name_embeddings = np.random.randn(len(unique_names), d_model) * 0.1

    samples, labels, metadata = [], [], []
    
    for row in rows:
        name = row['Pokemon_Name']
        name_vector = name_embeddings[name_to_id[name]]
        
        seq = [
            name_vector, 
            np.full(d_model, tier_map.get(row['Pokemon_Tier'], 0.1)),
            np.full(d_model, float(row['Level'])/100),
            np.full(d_model, float(row['HP_Percent'])/100),
            np.full(d_model, status_map.get(row['Status_Effect'], 0.0)),
            np.full(d_model, ball_map.get(row['Pokeball_Type'], 0.2))
        ]
        
        samples.append(np.array(seq))
        labels.append(1 if row['Catch_Result'] == 'Caught' else 0)
        metadata.append((name, row['Level'], row['HP_Percent'], row['Pokeball_Type']))
            
    return samples, labels, metadata

# --- 4. Main Execution ---
if __name__ == "__main__":
    CSV_FILE = 'raw_data.csv'
    
    try:
        X, y, meta = prepare_data_with_names(CSV_FILE, d_model=8)
        model = PokemonTransformerBlock(seq_len=6, d_model=8)
        
        print(f"--- 🔮 ผลการวิเคราะห์ด้วย Transformer (ทั้งหมด {len(X)} รายการ) ---")
        # พิมพ์หัวตาราง
        print(f"{'No.':<4} | {'Pokemon':<12} | {'Lv.':<4} | {'HP%':<4} | {'Actual':<8} | {'Predicted':<9} | {'Conf.':<6} | {'Correct'}")
        print("-" * 75)
        
        correct_predictions = 0
        
        # ลูปแสดงผลทุกรายการ
        for i in range(len(X)):
            prob, attn_weights = model.forward(X[i])
            pred = np.argmax(prob)
            
            p_name, p_lvl, p_hp, p_ball = meta[i]
            actual_str = 'Caught' if y[i] == 1 else 'Escaped'
            pred_str = 'Caught' if pred == 1 else 'Escaped'
            confidence = prob[0][pred]
            
            # เช็กว่าทายถูกหรือไม่
            is_correct = "✅" if actual_str == pred_str else "❌"
            if actual_str == pred_str:
                correct_predictions += 1
            
            # พิมพ์ทีละบรรทัดแบบจัดฟอร์แมต
            print(f"{i+1:<4} | {p_name:<12} | {p_lvl:<4} | {p_hp:<4} | {actual_str:<8} | {pred_str:<9} | {confidence:.0%} |   {is_correct}")
        
        # สรุปผลความแม่นยำตอนท้าย
        accuracy = correct_predictions / len(X)
        print("-" * 75)
        print(f"🎯 สรุปความแม่นยำรวม (Overall Accuracy): {accuracy:.2%} ({correct_predictions}/{len(X)})")
            
    except Exception as e:
        print(f"❌ พบข้อผิดพลาด: {e}")