import pandas as pd
import os
import torch
from transformers import pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"
OUTPUT_DIR = r"D:\Lyrics-Fanbase-Correlator\Final_Analysis_Results"

TARGET_ENDINGS = ['_MASTER.csv', '_COMMENTS.csv']

# map file prefixes to the exact names used in your FullDist files
# this prevents creating duplicates like Drizzy_FullDist.csv vs Drake_FullDist.csv
ARTIST_MAP = {
    "drizzy": "Drake", "drake": "Drake",
    "sabrinacarpenter": "SabrinaCarpenter", 
    "taylorswift": "TaylorSwift", 
    "eminem": "Eminem",
    "kanye": "KanyeWest", "kanyewest": "KanyeWest",
    "macmiller": "MacMiller",
    "playboicarti": "PlayboiCarti", "carti": "PlayboiCarti",
    "theweeknd": "TheWeeknd",
    "tameimpala": "TameImpala",
    "jcole": "J.Cole", 
    "kendricklamar": "KendrickLamar",
    "juicewrld": "JuiceWRLD", 
    "billieeilish": "BillieEilish", 
    "greenday": "GreenDay"
}

device = 0 if torch.cuda.is_available() else -1

print("Loading GoEmotions AI Model...")
classifier = pipeline(
    task="text-classification", 
    model="SamLowe/roberta-base-go_emotions", 
    top_k=None, 
    device=device,
    truncation=True,
    max_length=512
)

def get_standard_artist_name(filename):
    prefix = filename.split('_')[0].lower()
    return ARTIST_MAP.get(prefix, filename.split('_')[0])

def process_missing_files():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    files_to_process = []
    for f in os.listdir(INPUT_DIR):
        if any(f.endswith(ending) for ending in TARGET_ENDINGS):
            files_to_process.append(f)
            
    if not files_to_process:
        print("No MASTER or COMMENTS files found in the directory.")
        return

    for file_name in files_to_process:
        print(f"\n--- Processing {file_name} ---")
        artist_name = get_standard_artist_name(file_name)
        input_path = os.path.join(INPUT_DIR, file_name)
        
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue
            
        date_col = next((c for c in ['utc', 'created_utc', 'timestamp', 'Date'] if c in df.columns), None)
        text_col = next((c for c in ['text', 'body', 'selftext', 'Comment'] if c in df.columns), None)
        
        if not date_col or not text_col:
            print(f"Missing text or date column in {file_name}. Skipping.")
            continue
            
        # standardize dates so they match the existing FullDist format perfectly
        if pd.api.types.is_numeric_dtype(df[date_col]):
            df['Date'] = pd.to_datetime(df[date_col], unit='s', errors='coerce')
        else:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            
        df = df.dropna(subset=[text_col, 'Date'])
        df[text_col] = df[text_col].astype(str)
        
        # filter out the usual reddit junk
        df = df[~df[text_col].isin(['[removed]', '[deleted]', 'nan', ''])]
        
        if df.empty:
            print(f"No valid text rows left in {file_name}. Skipping.")
            continue
            
        print(f"Running AI on {len(df)} posts for {artist_name}...")
        
        texts = df[text_col].tolist()
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
        
        out_name = f"{artist_name}_FullDist.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        # safely append to the bottom of the existing file without overwriting
        # if the file doesn't exist yet, it will create it and write the headers
        file_exists = os.path.exists(out_path)
        final_df.to_csv(out_path, mode='a', header=not file_exists, index=False)
        
        print(f"Successfully appended {len(final_df)} new rows to {out_name}")

if __name__ == "__main__":
    process_missing_files()