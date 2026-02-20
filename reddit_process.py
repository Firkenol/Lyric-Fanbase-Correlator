import pandas as pd
from transformers import pipeline
import torch
import re
import os
from tqdm import tqdm

SOURCE_DIR = r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"
FINAL_OUTPUT_DIR = r"D:\Lyrics-Fanbase-Correlator\Final_Analysis_Results"
CHUNK_SIZE = 400

SUBREDDIT_MAP = {
    "taylorswift": "Taylor Swift", "sabrinacarpenter": "Sabrina Carpenter",
    "drizzy": "Drake", "kendricklamar": "Kendrick Lamar",
    "juicewrld": "Juice WRLD", "billieeilish": "Billie Eilish", "theweeknd": "The Weeknd",
    "greenday": "Green Day", "jcole": "J. Cole", "macmiller": "Mac Miller",
    "playboicarti": "Playboi Carti", "tameimpala": "Tame Impala"
}

def setup_classifier():
    device = 0 if torch.cuda.is_available() else -1
    print("Loading GoEmotions AI Model...")
    return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=device)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\[removed\]|\[deleted\]', '', text)
    text = re.sub(r'[^\w\s\.,!?\']', '', text)
    return text.strip()

def get_artist(filename):
    f = filename.lower()
    for sub, artist in SUBREDDIT_MAP.items():
        if sub in f: return artist
    return None

def main():
    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    classifier = setup_classifier()
    
    # strictly target filtered files only
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".csv") and "filtered" in f.lower()]
    artist_file_groups = {}
    
    for f in all_files:
        artist = get_artist(f)
        if artist:
            if artist not in artist_file_groups:
                artist_file_groups[artist] = []
            artist_file_groups[artist].append(f)

    for artist, files in artist_file_groups.items():
        print(f"\n--- ANALYZING: {artist} ---")
        out_path = os.path.join(FINAL_OUTPUT_DIR, f"{artist.replace(' ', '')}_FullDist.csv")
        
        merged_data = []
        for f in files:
            print(f"Reading {f}...")
            path = os.path.join(SOURCE_DIR, f)
            df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
            
            text_col = next((c for c in ['Text', 'body', 'selftext', 'body_text', 'comment_body', 'text'] if c in df.columns), None)
            date_col = next((c for c in ['Date', 'created_utc', 'timestamp', 'created', 'date'] if c in df.columns), None)

            if text_col and date_col:
                temp_df = df[[text_col, date_col]].copy()
                temp_df = temp_df.rename(columns={text_col: 'Text', date_col: 'Date'})
                
                # convert unix timestamps if present
                if df[date_col].dtype != 'object':
                    try:
                        temp_df['Date'] = pd.to_datetime(temp_df['Date'], unit='s')
                    except:
                        pass
                
                merged_data.append(temp_df)
            else:
                print(f"!!! WARNING: Skipped {f}. Could not find Text or Date columns. Found: {list(df.columns)}")
        
        if not merged_data: 
            print(f"!!! SKIP: No valid data found for {artist}")
            continue

        master_df = pd.concat(merged_data).dropna(subset=['Text', 'Date'])
        
        # remove duplicate entries
        original_count = len(master_df)
        master_df = master_df.drop_duplicates(subset=['Text'])
        new_count = len(master_df)
        if original_count != new_count:
            print(f"Dropped {original_count - new_count} duplicate posts.")
        
        start_row = 0
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, usecols=['Date'])
            start_row = len(existing)
            print(f"Resuming {artist} from row {start_row}...")

        total_rows = len(master_df)
        if start_row >= total_rows:
            print(f"Already finished {artist}. Skipping.")
            continue

        for i in tqdm(range(start_row, total_rows, CHUNK_SIZE), desc=f"Processing {artist}"):
            chunk = master_df.iloc[i : i + CHUNK_SIZE].copy()
            chunk['Clean_Text'] = chunk['Text'].apply(clean_text)
            chunk = chunk[chunk['Clean_Text'] != ""]
            if chunk.empty: continue
            
            results = classifier(chunk['Clean_Text'].tolist(), truncation=True, max_length=512)
            dist_list = [{item['label']: item['score'] for item in res} for res in results]
            dist_df = pd.DataFrame(dist_list).set_index(chunk.index)
            
            final_chunk = pd.concat([chunk, dist_df], axis=1)
            final_chunk.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)

if __name__ == "__main__":
    main()