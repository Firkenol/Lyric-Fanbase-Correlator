import json
import os
import pandas as pd
from datetime import datetime

RAW_DIR = r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"

# unix timestamps for 1 month before and after the album drops
# kanye mbdtf: nov 22 2010
KANYE_START = 1287705600 
KANYE_END = 1293062400   

# mac blue slide park: nov 8 2011
MAC_START = 1318032000 
MAC_END = 1323388800   

TARGETS = {
    "Kanye_comments": (KANYE_START, KANYE_END, "Kanye_Filtered.csv"),
    "Kanye_submissions": (KANYE_START, KANYE_END, "Kanye_Filtered.csv"),
    "MacMiller_comments": (MAC_START, MAC_END, "MacMiller_Filtered.csv"),
    "MacMiller_submissions": (MAC_START, MAC_END, "MacMiller_Filtered.csv")
}

def scan_and_append():
    for raw_file, (start_utc, end_utc, target_file) in TARGETS.items():
        input_path = os.path.join(RAW_DIR, raw_file)
        target_path = os.path.join(RAW_DIR, target_file)
        
        if not os.path.exists(input_path):
            print(f"skipping {raw_file}, file not found")
            continue
            
        if not os.path.exists(target_path):
            print(f"target file {target_file} not found. cannot append to it. skipping.")
            continue
            
        # read the exact columns from the target file so we don't misalign data
        target_df = pd.read_csv(target_path, nrows=0)
        target_columns = target_df.columns.tolist()
        
        # lower case columns for flexible matching
        col_map_lower = {c.lower(): c for c in target_columns}
        
        print(f"scanning {raw_file} for valid 2010/2011 dates...")
        valid_records = []
        
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    utc_val = data.get('created_utc')
                    
                    if utc_val:
                        utc_int = int(utc_val)
                        
                        # if the post falls exactly in the album hype window, save it
                        if start_utc <= utc_int <= end_utc:
                            text = data.get('body') or data.get('selftext') or data.get('title') or ""
                            author = data.get('author', '[deleted]')
                            score = data.get('score', 0)
                            
                            if text not in ['[removed]', '[deleted]', '']:
                                row_data = {}
                                
                                # dynamically map the parsed data to the existing csv columns
                                date_str = datetime.utcfromtimestamp(utc_int).strftime('%Y-%m-%d %H:%M:%S')
                                if 'date' in col_map_lower: 
                                    row_data[col_map_lower['date']] = date_str
                                elif 'utc' in col_map_lower: 
                                    row_data[col_map_lower['utc']] = utc_int
                                elif 'created_utc' in col_map_lower: 
                                    row_data[col_map_lower['created_utc']] = utc_int
                                    
                                if 'text' in col_map_lower: 
                                    row_data[col_map_lower['text']] = text
                                elif 'body' in col_map_lower: 
                                    row_data[col_map_lower['body']] = text
                                    
                                if 'score' in col_map_lower: 
                                    row_data[col_map_lower['score']] = score
                                if 'author' in col_map_lower: 
                                    row_data[col_map_lower['author']] = author
                                    
                                valid_records.append(row_data)
                except Exception:
                    continue
        
        if valid_records:
            append_df = pd.DataFrame(valid_records)
            
            # ensure column order matches target exactly
            for col in target_columns:
                if col not in append_df.columns:
                    append_df[col] = None
            append_df = append_df[target_columns]
            
            # append directly to the target file
            append_df.to_csv(target_path, mode='a', header=False, index=False)
            print(f"SUCCESS: appended {len(valid_records)} posts from {raw_file} to {target_file}")
        else:
            print(f"ghost town confirmed. 0 posts found in {raw_file} for that window.")

if __name__ == "__main__":
    scan_and_append()