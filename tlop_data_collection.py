import json
import os

RAW_DIR = r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"

# the life of pablo: feb 14, 2016
# window: jan 31 to feb 28
TLOP_START = 1454198400 
TLOP_END = 1456617600

TARGETS = [
    ("Kanye_comments", TLOP_START, TLOP_END, "The Life of Pablo"),
    ("Kanye_submissions", TLOP_START, TLOP_END, "The Life of Pablo")
]

def check_kanye_volume():
    for raw_file, start_utc, end_utc, album_name in TARGETS:
        input_path = os.path.join(RAW_DIR, raw_file)
        if not os.path.exists(input_path):
            continue
            
        found_count = 0
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    utc = data.get('created_utc')
                    if utc and start_utc <= int(utc) <= end_utc:
                        text = data.get('body') or data.get('selftext') or data.get('title') or ""
                        if text not in ['[removed]', '[deleted]', '']:
                            found_count += 1
                except Exception:
                    continue
        print(f"{album_name} ({raw_file}): Found {found_count} valid posts")

if __name__ == "__main__":
    check_kanye_volume()