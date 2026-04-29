import pandas as pd
import random

def main():
    # ==========================================
    # 1. กำหนดตัวเลือกของแต่ละ Input และรายชื่อโปเกมอน
    # ==========================================
    tiers = ['Common', 'Rare', 'Legendary']
    statuses = ['None', 'Sleep', 'Paralyze', 'Poison']
    pokeballs = ['Poke Ball', 'Great Ball', 'Ultra Ball']

    # รายชื่อโปเกมอนแบ่งตามระดับความแรร์
    pokemon_names = {
        'Common': ['Pidgey', 'Rattata', 'Caterpie', 'Weedle', 'Zubat', 'Magikarp', 'Bidoof', 'Sentret', 'Zigzagoon', 'Geodude', 'Oddish', 'Poliwag'],
        'Rare': ['Snorlax', 'Lapras', 'Dratini', 'Larvitar', 'Bagon', 'Beldum', 'Gible', 'Chansey', 'Aerodactyl', 'Scyther', 'Gyarados', 'Arcanine'],
        'Legendary': ['Mewtwo', 'Lugia', 'Rayquaza', 'Dialga', 'Palkia', 'Giratina', 'Arceus', 'Ho-Oh', 'Groudon', 'Kyogre', 'Zapdos', 'Articuno']
    }

    data = []

    print("⏳ กำลังสร้างชุดข้อมูลการจับโปเกมอน 5,000 เรคคอร์ด (พร้อมชื่อ)...")

    # ==========================================
    # 2. วนลูปสร้างข้อมูล 5,000 เรคคอร์ด
    # ==========================================
    for _ in range(5000):
        # สุ่มระดับความแรร์
        tier = random.choices(tiers, weights=[0.6, 0.3, 0.1])[0]
        
        # สุ่มชื่อโปเกมอนให้ตรงกับความแรร์
        name = random.choice(pokemon_names[tier])
        
        # สุ่มค่าสถานะอื่นๆ
        level = random.randint(1, 100)
        hp = random.randint(1, 100)
        status = random.choices(statuses, weights=[0.5, 0.2, 0.2, 0.1])[0]
        ball = random.choice(pokeballs)
        
        # --- ตรรกะการคำนวณโอกาสจับได้ (Catch Rate Logic) ---
        catch_score = 0
        
        if ball == 'Ultra Ball': catch_score += 40
        elif ball == 'Great Ball': catch_score += 20
        else: catch_score += 10
            
        if status in ['Sleep', 'Paralyze']: catch_score += 25
        elif status == 'Poison': catch_score += 10
            
        if hp <= 20: catch_score += 30
        elif hp <= 50: catch_score += 15
            
        if tier == 'Legendary': catch_score -= 60
        elif tier == 'Rare': catch_score -= 20
            
        catch_score -= (level * 0.3) 
        
        rng = random.randint(-10, 20)
        final_score = catch_score + rng
        
        if final_score > 40:
            result = 'Caught'
        else:
            result = 'Escaped'
            
        # เก็บข้อมูล
        data.append([name, tier, level, hp, status, ball, result])

    # ==========================================
    # 3. สร้างตารางและบันทึกเป็นไฟล์ CSV
    # ==========================================
    df = pd.DataFrame(data, columns=['Pokemon_Name', 'Pokemon_Tier', 'Level', 'HP_Percent', 'Status_Effect', 'Pokeball_Type', 'Catch_Result'])

    file_name = 'pokemon_catch_data_with_names.csv'
    df.to_csv(file_name, index=False, encoding='utf-8')

    print(f"✅ สร้างไฟล์ '{file_name}' สำเร็จเรียบร้อยแล้ว!")
    print(f"📁 ไฟล์ถูกบันทึกไว้ที่โฟลเดอร์เดียวกับไฟล์ .py นี้")
    print("\n--- ตัวอย่างข้อมูล 5 แถวแรก ---")
    print(df.head().to_string())

if __name__ == "__main__":
    main()