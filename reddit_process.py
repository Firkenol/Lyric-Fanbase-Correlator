import pandas as pd
from transformers import pipeline
import torch
import re
import os
from tqdm import tqdm

# --- ONE FOLDER TO RULE THEM ALL ---
# This is where you said you moved all the "old shit" and new raw files
SOURCE_DIR = r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"
FINAL_OUTPUT_DIR = r"D:\Lyrics-Fanbase-Correlator\Final_Analysis_Results"
CHUNK_SIZE = 400

# Subreddit names to look for in the filenames
SUBREDDIT_MAP = {
    "eminem": "Eminem", "taylorswift": "Taylor Swift", "sabrinacarpenter": "Sabrina Carpenter",
    "kanye": "Kanye West", "drizzy": "Drake", "kendricklamar": "Kendrick Lamar",
    "juicewrld": "Juice WRLD", "billieeilish": "Billie Eilish", "theweeknd": "The Weeknd",
    "greenday": "Green Day", "jcole": "J. Cole", "macmiller": "Mac Miller",
    "playboicarti": "Playboi Carti", "tameimpala": "Tame Impala"
}

def setup_classifier():
    # Detects if you have an Nvidia GPU, otherwise uses CPU
    device = 0 if torch.cuda.is_available() else -1
    print("Loading GoEmotions AI Model...")
    return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=device)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\[removed\]|\[deleted\]', '', text)
    text = re.sub(r'[^\w\s\.,!?\']', '', text) # keep basic punctuation for context
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
    
    # Scan the folder and group files by artist
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".csv")]
    artist_file_groups = {}
    
    for f in all_files:
        artist = get_artist(f)
        if artist:
            if artist not in artist_file_groups:
                artist_file_groups[artist] = []
            artist_file_groups[artist].append(f)

    for artist, files in artist_file_groups.items():
        print(f"\n--- ANALYZING: {artist} ({len(files)} files found) ---")
        
        out_path = os.path.join(FINAL_OUTPUT_DIR, f"{artist.replace(' ', '')}_FullDist.csv")
        
        # Load and combine all data for this artist first
        merged_data = []
        for f in files:
            print(f"Reading {f}...")
            path = os.path.join(SOURCE_DIR, f)
            df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
            
            # Find the text column (handles 'Text', 'body', 'selftext')
            text_col = next((c for c in ['Text', 'body', 'selftext'] if c in df.columns), None)
            if text_col:
                df = df.rename(columns={text_col: 'Text'})
                # Keep only what we need to save memory
                merged_data.append(df[['Text', 'Date']])
        
        if not merged_data: continue
        master_df = pd.concat(merged_data).dropna(subset=['Text', 'Date'])
        
        # Check for checkpoint: Skip rows we already analyzed
        start_row = 0
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, usecols=['Date'])
            start_row = len(existing)
            print(f"Resuming {artist} from row {start_row}...")

        # Processing the combined dataset in chunks
        total_rows = len(master_df)
        for i in tqdm(range(start_row, total_rows, CHUNK_SIZE), desc=f"GoEmotions Progress ({artist})"):
            chunk = master_df.iloc[i : i + CHUNK_SIZE].copy()
            
            # Clean and Clasify
            chunk['Clean_Text'] = chunk['Text'].apply(clean_text)
            chunk = chunk[chunk['Clean_Text'] != ""]
            if chunk.empty: continue
            
            texts = chunk['Clean_Text'].tolist()
            results = classifier(texts, truncation=True, max_length=512)
            
            # Create a dataframe for all 28 emotion scores
            dist_list = [{item['label']: item['score'] for item in res} for res in results]
            dist_df = pd.DataFrame(dist_list).set_index(chunk.index)
            
            final_chunk = pd.concat([chunk, dist_df], axis=1)
            final_chunk['Date'] = pd.to_datetime(final_chunk['Date'], errors='coerce')
            
            # Save progress
            write_head = not os.path.exists(out_path)
            final_chunk.to_csv(out_path, mode='a', header=write_head, index=False)

if __name__ == "__main__":
    main()