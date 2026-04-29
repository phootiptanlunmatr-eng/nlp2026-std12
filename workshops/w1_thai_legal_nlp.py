import os
import re
import csv
import numpy as np
import collections
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

# =============================================================================
# SECTION 0: Dataset — โหลดข้อมูลจาก CSV (Pokemon Catching Data)
# =============================================================================
BASE_DIR = 'D:\\nlp2026\\workshops\\'
DATA_PATH = os.path.join(BASE_DIR, 'raw_data.csv')

def load_pokemon_corpus(file_path: str) -> Tuple[List[str], Dict]:
    """
    โหลดข้อมูลจาก CSV และแปลงเป็นข้อความ (String Representation) เพื่อใช้ใน Pipeline
    Returns: (list of texts, metadata dict)
    """
    if not os.path.exists(file_path):
        print(f"❌ ไม่พบไฟล์ที่: {file_path} ใช้ข้อมูล Fallback")
        return [
            "Magikarp, Common, Level 24, 71% HP, Status: Paralyze, Ball: Poke Ball",
            "Rayquaza, Legendary, Level 86, 12% HP, Status: None, Ball: Poke Ball",
        ], {"source": "fallback"}

    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # สร้างประโยคอธิบายข้อมูล Pokemon เพื่อจำลอง NLP Task
            text = f"{row['Pokemon_Name']}, {row['Pokemon_Tier']}, Level {row['Level']}, {row['HP_Percent']}% HP, Status: {row['Status_Effect']}, Ball: {row['Pokeball_Type']}"
            texts.append(text)

    return texts, {"source": "csv", "count": len(texts)}

# Load data
POKEMON_CORPUS, CORPUS_META = load_pokemon_corpus(DATA_PATH)

# =============================================================================
# SECTION 1: Tokenizer — ปรับปรุงให้รองรับข้อมูล Pokemon
# =============================================================================
class PokemonTokenizer:
    def __init__(self):
        # กำหนดคำสำคัญที่ต้องการแยกเป็นพิเศษ
        self.special_tokens = {"Legendary", "Rare", "Common", "Ultra Ball", "Great Ball", "Poke Ball"}
        
    def tokenize(self, text: str) -> List[str]:
        # ล้างคำที่ไม่เกี่ยวข้องและแยกด้วย comma หรือ space
        text = text.replace("% HP", " HP")
        tokens = re.split(r'[,\s]+', text)
        return [t for t in tokens if t]

# =============================================================================
# SECTION 2: Feature Extractor (TF-IDF Style)
# =============================================================================
class PokemonFeatureExtractor:
    def __init__(self, max_features: int = 100):
        self.tokenizer = PokemonTokenizer()
        self.max_features = max_features
        self.vocab = {}
        self.idf = {}

    def fit_transform(self, docs: List[str]) -> np.ndarray:
        all_tokens = [self.tokenizer.tokenize(d) for d in docs]
        
        # Build Vocab
        counts = collections.Counter([t for tokens in all_tokens for t in tokens])
        common = counts.most_common(self.max_features)
        self.vocab = {word: i for i, (word, _) in enumerate(common)}
        
        # Compute TF-IDF
        X = np.zeros((len(docs), len(self.vocab)))
        for i, tokens in enumerate(all_tokens):
            t_counts = collections.Counter(tokens)
            for word, count in t_counts.items():
                if word in self.vocab:
                    X[i, self.vocab[word]] = count
        return X

    def get_top_terms(self, vec: np.ndarray, n: int = 5) -> List[Tuple[str, float]]:
        inv_vocab = {v: k for k, v in self.vocab.items()}
        indices = np.argsort(vec)[-n:][::-1]
        return [(inv_vocab[i], vec[i]) for i in indices if vec[i] > 0]

# =============================================================================
# SECTION 3: Entity Extractor — สกัดข้อมูลสำคัญ (Tier, Status, Ball)
# =============================================================================
@dataclass
class PokemonEntity:
    entity_type: str  # Tier, Status, Ball, Level
    value: str
    confidence: float
    context_signals: List[str] = field(default_factory=list)

class PokemonInfoExtractor:
    def __init__(self):
        self.patterns = {
            "Tier": r"(Legendary|Rare|Common)",
            "Status": r"Status:\s*(\w+)",
            "Ball": r"Ball:\s*([\w\s]+Ball)",
            "Level": r"Level\s*(\d+)"
        }

    def extract(self, text: str) -> List[PokemonEntity]:
        entities = []
        for etype, pat in self.patterns.items():
            match = re.search(pat, text)
            if match:
                val = match.group(1)
                conf = 0.95 if etype in ["Tier", "Level"] else 0.80
                signals = ["explicit_mention"]
                entities.append(PokemonEntity(etype, val, conf, signals))
        return entities

# =============================================================================
# SECTION 4: Tier Hierarchy — คำนวณน้ำหนักความยากในการจับ (Capture Weight)
# =============================================================================
class PokemonTierHierarchy:
    def __init__(self):
        # กำหนดลำดับความสำคัญ/ความยาก (คล้ายกับ Legal Hierarchy ในโค้ดเดิม)
        self.tier_weights = {
            "Legendary": 10.0,
            "Rare": 5.0,
            "Common": 1.0
        }

    def compute_capture_difficulty(self, entities: List[PokemonEntity]) -> float:
        weight = 0.0
        for e in entities:
            if e.entity_type == "Tier" and e.value in self.tier_weights:
                weight = self.tier_weights[e.value]
        return weight

# =============================================================================
# SECTION 5: Pipeline — รวมขั้นตอนทั้งหมด
# =============================================================================
class PokemonPipeline:
    def __init__(self, max_features: int = 50):
        self.fe = PokemonFeatureExtractor(max_features=max_features)
        self.extractor = PokemonInfoExtractor()
        self.hierarchy = PokemonTierHierarchy()

    def run_analysis(self, corpus: List[str]):
        print(f"--- Pokemon Catch Analysis Pipeline ---")
        X = self.fe.fit_transform(corpus)
        
        for i, doc in enumerate(corpus[:5]): # ดูตัวอย่าง 5 ตัวแรก
            entities = self.extractor.extract(doc)
            difficulty = self.hierarchy.compute_capture_difficulty(entities)
            
            print(f"\nSample {i+1}: {doc}")
            print(f"  Detected Entities: {[(e.entity_type, e.value) for e in entities]}")
            print(f"  Capture Difficulty Score: {difficulty}/10.0")
            
            top_terms = self.fe.get_top_terms(X[i], n=3)
            print(f"  Top Features: {[t for t, _ in top_terms]}")

# =============================================================================
# MAIN EXECUTION (Smoke Test)
# =============================================================================
if __name__ == "__main__":
    pipeline = PokemonPipeline()
    pipeline.run_analysis(POKEMON_CORPUS)