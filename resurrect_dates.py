import pandas as pd
import os
import gc
import warnings
warnings.filterwarnings('ignore')

DIR = r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"

def resurrect_unix_times():
    print("Building global hashmap of Unix timestamps...")
    master_date_map = {}
    
    # scan every file to build the dictionary
    for f in os.listdir(DIR):
        if f.endswith("_Filtered.csv") or f.endswith("_COMMENTS.csv"):
            path = os.path.join(DIR, f)
            try:
                df = pd.read_csv(path, low_memory=False, on_bad_lines='skip')
                
                cols_lower = {str(c).lower().strip(): c for c in df.columns}
                text_col = next((cols_lower[c] for c in ['comment', 'body', 'selftext', 'title', 'text', 'content'] if c in cols_lower), None)
                date_col = next((cols_lower[c] for c in ['created_utc', 'timestamp', 'date', 'utc'] if c in cols_lower), None)
                
                if text_col and date_col:
                    temp_map = df.dropna(subset=[text_col, date_col]).drop_duplicates(subset=[text_col]).set_index(text_col)[date_col].to_dict()
                    master_date_map.update(temp_map)
                    print(f"Grabbed {len(temp_map)} timestamps from {f}")
                    
                del df
                gc.collect()
            except Exception:
                continue

    print(f"\nTotal unique Unix timestamps mapped: {len(master_date_map)}")
    
    if not master_date_map:
        print("No timestamps found in the source files. Exiting.")
        return

    # inject them into the MASTER files
    for f in os.listdir(DIR):
        if f.endswith("_MASTER.csv"):
            path = os.path.join(DIR, f)
            print(f"\nInjecting Unix times into {f}...")
            
            try:
                df_mast = pd.read_csv(path, low_memory=False, on_bad_lines='skip')
                
                mast_cols_lower = {str(c).lower().strip(): c for c in df_mast.columns}
                mast_text_col = next((mast_cols_lower[c] for c in ['comment', 'body', 'selftext', 'title', 'text', 'content'] if c in mast_cols_lower), None)
                
                if not mast_text_col:
                    print(f"No text column found in {f}.")
                    continue
                    
                df_mast['created_utc'] = df_mast[mast_text_col].map(master_date_map)
                
                matched = df_mast['created_utc'].notna().sum()
                print(f"Resurrected {matched} / {len(df_mast)} timestamps.")
                
                df_mast.to_csv(path, index=False)
                del df_mast
                gc.collect()
            except Exception as e:
                print(f"Failed to process {f} - {e}")

if __name__ == "__main__":
    resurrect_unix_times()