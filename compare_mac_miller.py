import json
import os

RAW_DIR = r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"

# watching movies with the sound off: june 18 2013
# window: may 18 to july 18
WMWTSO_START = 1368835200
WMWTSO_END = 1374105600

# swimming: aug 3 2018
# window: july 3 to sept 3
SWIMMING_START = 1530576000
SWIMMING_END = 1535932800

TARGETS = [
    ("MacMiller_comments", WMWTSO_START, WMWTSO_END, "WMWTSO"),
    ("MacMiller_submissions", WMWTSO_START, WMWTSO_END, "WMWTSO"),
    ("MacMiller_comments", SWIMMING_START, SWIMMING_END, "Swimming"),
    ("MacMiller_submissions", SWIMMING_START, SWIMMING_END, "Swimming")
]

def check_album_volume():
    for raw_file, start_utc, end_utc, album_name in TARGETS:
        input_path = os.path.join(RAW_DIR, raw_file)
        
        if not os.path.exists(input_path):
            continue
            
        found_count = 0
        
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    utc_val = data.get('created_utc')
                    
                    if utc_val:
                        utc_int = int(utc_val)
                        if start_utc <= utc_int <= end_utc:
                            text = data.get('body') or data.get('selftext') or data.get('title') or ""
                            if text not in ['[removed]', '[deleted]', '']:
                                found_count += 1
                except Exception:
                    continue
        
        print(f"{album_name} ({raw_file}): Found {found_count} valid posts")

if __name__ == "__main__":
    check_album_volume()