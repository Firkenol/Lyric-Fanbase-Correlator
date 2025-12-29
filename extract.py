import zstandard as zstd
import json
import csv
import os
import io
from datetime import datetime

# --- DIR SETUP ---
# Path to the raw reddit dumps on the drive
raw_path = r"D:\Lyrics-Fanbase-Correlator\Lyric-Fanbase-Correlator\raw"
# Where the cleaned CSVs will end up
out_path = r"D:\Lyrics-Fanbase-Correlator\Lyric-Fanbase-Correlator\processed"

# My timeframe: 2024 through end of 2025 (Unix timestamps)
start_t = 1704067200 
end_t = 1767225600 

if not os.path.exists(out_path):
    os.makedirs(out_path)

def extract(file_name):
    in_file = os.path.join(raw_path, file_name)
    out_file = os.path.join(out_path, f"{file_name}_clean.csv")
    
    if not os.path.exists(in_file):
        print(f"! Missing: {file_name}")
        return

    print(f">> Opening {file_name}...")
    
    count = 0
    errors = 0
    lines_seen = 0

    with open(in_file, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            # errors='ignore' handles the weird encoding glitches in reddit text
            stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')
            
            with open(out_file, 'w', newline='', encoding='utf-8') as csv_out:
                w = csv.writer(csv_out)
                w.writerow(['utc', 'date', 'text', 'score'])
                
                for line in stream:
                    lines_seen += 1
                    try:
                        data = json.loads(line)
                        ts = int(data.get('created_utc', 0))
                        
                        # Only grab the data for our research years
                        if start_t <= ts <= end_t:
                            # Catching both posts (title+body) and comments (body)
                            body = data.get('body', '')
                            title = data.get('title', '')
                            full_text = f"{title} {body}".strip()
                            
                            # Flattening the text so newlines don't break the CSV
                            clean_txt = " ".join(full_text.split())
                            date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                            
                            w.writerow([ts, date_str, clean_txt, data.get('score', 0)])
                            count += 1
                            
                    except Exception:
                        # This skips the bad or corrupted lines
                        errors += 1
                        continue
                    
                    # Heartbeat so I know it's not frozen
                    if lines_seen % 200000 == 0:
                        print(f"   ... through {lines_seen} lines, kept {count}")

    print(f"Done with {file_name}. Kept: {count} | Skipped errors: {errors}\n")

#List of artists
subs = [
    "Eminem_submissions", "Eminem_comments",
    "TaylorSwift_submissions", "TaylorSwift_comments",
    "SabrinaCarpenter_submissions", "SabrinaCarpenter_comments",
    "Kanye_submissions", "Kanye_comments",
    "drizzy_submissions", "drizzy_comments",
    "KendrickLamar_submissions", "KendrickLamar_comments",
    "JuiceWRLD_submissions", "JuiceWRLD_comments",
    "billieeilish_submissions", "billieeilish_comments",
    "TheWeeknd_submissions", "TheWeeknd_comments",
    "greenday_submissions", "greenday_comments",
    "Jcole_submissions", "Jcole_comments",
    "MacMiller_submissions", "MacMiller_comments",
    "playboicarti_submissions", "playboicarti_comments",
    "TameImpala_submissions", "TameImpala_comments"
]

if __name__ == "__main__":
    for s in subs:
        extract(s)
    print("All artists processed. Check your processed folder.")