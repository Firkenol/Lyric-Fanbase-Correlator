import json
import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"
OUTPUT_DIR = r"D:\Lyrics-Fanbase-Correlator\Final_Analysis_Results"

# windows: 1 month before/after
# The Life of Pablo: feb 14 2016
TLOP_START = 1454198400 
TLOP_END = 1456617600 

# swimming: aug 3 2018
S_START = 1530576000 
S_END = 1535932800   

TARGETS = [
    ("Kanye_comments", TLOP_START, TLOP_END, "KanyeWest"),
    ("Kanye_submissions", TLOP_START, TLOP_END, "KanyeWest"),
    ("MacMiller_comments", S_START, S_END, "MacMiller"),
    ("MacMiller_submissions", S_START, S_END, "MacMiller")
]

device = 0 if torch.cuda.is_available() else -1

def process_legacy_data():
    print("Loading GoEmotions AI Model...")
    classifier = pipeline(
        task="text-classification", 
        model="SamLowe/roberta-base-go_emotions", 
        top_k=None, 
        device=device,
        truncation=True,
        max_length=512
    )
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for raw_file, start_utc, end_utc, artist in TARGETS:
        input_path = os.path.join(RAW_DIR, raw_file)
        if not os.path.exists(input_path):
            print(f"Skipping {raw_file} - not found.")
            continue
            
        print(f"\n--- Extracting {artist} from {raw_file} ---")
        extracted = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    utc = data.get('created_utc')
                    if utc and start_utc <= int(utc) <= end_utc:
                        text = data.get('body') or data.get('selftext') or data.get('title') or ""
                        if text not in ['[removed]', '[deleted]', '']:
                            # format identically to your _FullDist files
                            date_str = datetime.utcfromtimestamp(int(utc)).strftime('%Y-%m-%d %H:%M:%S')
                            extracted.append({'Date': date_str, 'Comment': text})
                except Exception:
                    continue
                    
        if not extracted:
            print("No valid posts found in window.")
            continue
            
        df = pd.DataFrame(extracted)
        print(f"Found {len(df)} posts. Running AI analysis...")
        
        texts = df['Comment'].tolist()
        batch_size = 100
        all_results = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            results = classifier(batch)
            for res in results:
                emotion_dict = {score['label']: score['score'] for score in res}
                all_results.append(emotion_dict)
                
        emotions_df = pd.DataFrame(all_results)
        final_df = pd.concat([df.reset_index(drop=True), emotions_df.reset_index(drop=True)], axis=1)
        
        out_path = os.path.join(OUTPUT_DIR, f"{artist}_FullDist.csv")
        file_exists = os.path.exists(out_path)
        
        final_df.to_csv(out_path, mode='a', header=not file_exists, index=False)
        print(f"Appended {len(final_df)} analyzed rows to {artist}_FullDist.csv")

if __name__ == "__main__":
    process_legacy_data()