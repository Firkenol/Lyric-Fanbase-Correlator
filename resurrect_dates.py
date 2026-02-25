import pandas as pd
import os
import gc

DIR = r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"

FILE_PAIRS = [
    ("Playboi Carti_Filtered.csv", "PlayboiCarti_MASTER.csv"),
    ("The Weeknd_Filtered.csv", "TheWeeknd_MASTER.csv"),
    ("Drake_Filtered.csv", "Drake_MASTER.csv"),
    ("Kanye_Filtered.csv", "Kanye_MASTER.csv")
]

def resurrect_timestamps():
    for filtered_file, master_file in FILE_PAIRS:
        filt_path = os.path.join(DIR, filtered_file)
        mast_path = os.path.join(DIR, master_file)
        
        if not os.path.exists(filt_path) or not os.path.exists(mast_path):
            continue
            
        print(f"\nBuilding hashmap from {filtered_file}...")
        df_filt = pd.read_csv(filt_path, low_memory=False, on_bad_lines='skip')
        
        # print the exact columns so we aren't blind
        print(f"  Columns found: {list(df_filt.columns)}")
        
        # force lowercase and strip spaces to brute-force the match
        cols_lower = {str(c).lower().strip(): c for c in df_filt.columns}
        
        text_col = None
        for cand in ['comment', 'body', 'selftext', 'title', 'text', 'content']:
            if cand in cols_lower:
                text_col = cols_lower[cand]
                break
                
        date_col = None
        for cand in ['date', 'created_utc', 'timestamp', 'created']:
            if cand in cols_lower:
                date_col = cols_lower[cand]
                break
        
        if not text_col or not date_col:
            print(f"  [!] Missing text or date column in {filtered_file}. Skipping.")
            continue
            
        print(f"  Mapped text to '{text_col}' and date to '{date_col}'")
        
        date_map = df_filt.dropna(subset=[text_col]).drop_duplicates(subset=[text_col]).set_index(text_col)[date_col].to_dict()
        
        del df_filt
        gc.collect()
        
        print(f"Applying hashmap to {master_file}...")
        df_mast = pd.read_csv(mast_path, low_memory=False, on_bad_lines='skip')
        
        mast_cols_lower = {str(c).lower().strip(): c for c in df_mast.columns}
        mast_text_col = None
        for cand in ['comment', 'body', 'selftext', 'title', 'text', 'content']:
            if cand in mast_cols_lower:
                mast_text_col = mast_cols_lower[cand]
                break
                
        if not mast_text_col:
            print(f"  [!] Could not find text column in {master_file} to map against.")
            continue
            
        df_mast['Date'] = df_mast[mast_text_col].map(date_map)
        
        matched = df_mast['Date'].notna().sum()
        print(f"  Resurrected {matched} / {len(df_mast)} timestamps.")
        
        # overwrite the master file with the new dates included
        df_mast.to_csv(mast_path, index=False)

if __name__ == "__main__":
    resurrect_timestamps()