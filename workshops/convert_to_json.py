"""
convert_to_json.py (v3 — Professional Edition)
------------------
อ่าน data/raw/thai_ip_corpus.txt → สร้าง data/processed/thai_ip_corpus.json
พร้อม Silver Standard metadata, Context-aware Confidence, และ Ambiguity Flagging

Changelog v3 (15 April 2026):
    [v3-1] เพิ่ม Context-aware Confidence Scoring
    [v3-2] เพิ่ม Legal Hierarchy Classification
    [v3-3] เพิ่ม Ambiguity Flagging
    [v3-4] เพิ่ม Physics Gate Weight Preview
"""
import os
import csv
import json
import numpy as np
from datetime import datetime

# ตั้งค่า Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'raw_data.csv')
JSON_OUT_PATH = os.path.join(BASE_DIR, 'pokemon_processed_data.json')

def calculate_silver_metadata(row):
    """คำนวณ Metadata เพื่อจำลองระบบ Expert Review ในงาน NLP"""
    hp = float(row['HP_Percent'])
    tier = row['Pokemon_Tier']
    ball = row['Pokeball_Type']
    result = row['Catch_Result']
    
    # 1. Confidence Score (ความน่าเชื่อถือของข้อมูล)
    # เช่น ถ้า HP น้อย + ใช้ Ultra Ball + จับได้ = มั่นใจสูง
    confidence = 0.5
    if result == "Caught" and hp < 20: confidence += 0.3
    if ball == "Ultra Ball": confidence += 0.2
    
    # 2. Ambiguity Flag (ความผิดปกติ/กำกวม)
    # เช่น Legendary HP 100% แต่จับได้ด้วย Poke Ball (อาจเป็น Outlier)
    is_ambiguous = False
    if tier == "Legendary" and hp > 90 and result == "Caught" and ball == "Poke Ball":
        is_ambiguous = True
        
    return round(min(confidence, 1.0), 2), is_ambiguous

def convert_csv_to_json():
    samples = []
    print(f"🔄 กำลังแปลงไฟล์: {CSV_PATH}...")

    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            conf, ambig = calculate_silver_metadata(row)
            
            # สร้างข้อความ Text Representation สำหรับ NLP Task
            text_desc = (f"พบ {row['Pokemon_Name']} ระดับ {row['Pokemon_Tier']} "
                         f"เลเวล {row['Level']} มี HP {row['HP_Percent']}% "
                         f"สถานะ {row['Status_Effect']} ใช้ {row['Pokeball_Type']}")
            
            sample = {
                "id": i,
                "text": text_desc,
                "label": 1 if row['Catch_Result'] == "Caught" else 0,
                "metadata": {
                    "tier": row['Pokemon_Tier'],
                    "confidence": conf,
                    "is_ambiguous": ambig,
                    "timestamp": datetime.now().isoformat()
                }
            }
            samples.append(sample)

    with open(JSON_OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"data": samples, "version": "1.0-pokemon"}, f, ensure_ascii=False, indent=2)
    
    print(f"✅ สร้างไฟล์สำเร็จ: {JSON_OUT_PATH} (รวม {len(samples)} รายการ)")

if __name__ == "__main__":
    convert_csv_to_json()