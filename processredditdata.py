import json
import csv
import datetime
import re
import os
import emoji
from tqdm import tqdm


INPUT_DIRECTORY = r"D:\Lyrics-Fanbase-Correlator\Lyric-Fanbase-Correlator\raw"

CSV_SOURCES = {
    "The Weeknd": [r"D:\Lyrics-Fanbase-Correlator\Lyric-Fanbase-Correlator\Processed_Reddit_Data\TheWeeknd_MASTER.csv"],
    "Drake": [r"D:\Lyrics-Fanbase-Correlator\Lyric-Fanbase-Correlator\Processed_Reddit_Data\Drizzy_MASTER.csv"],
    "Mac Miller": [r"D:\Lyrics-Fanbase-Correlator\Lyric-Fanbase-Correlator\Processed_Reddit_Data\MacMiller_MASTER.csv"],
    "Playboi Carti": [r"D:\Lyrics-Fanbase-Correlator\Lyric-Fanbase-Correlator\Processed_Reddit_Data\PlayboiCarti_MASTER.csv"],
    "Sabrina Carpenter": [r"D:\Lyrics-Fanbase-Correlator\Lyric-Fanbase-Correlator\Processed_Reddit_Data\SabrinaCarpenter_COMMENTS.csv"],
    "TameImpala": [r"D:\Lyrics-Fanbase-Correlator\Lyric-Fanbase-Correlator\Processed_Reddit_Data\TameImpala_COMMENTS.csv"],
    "Taylor Swift": [r"D:\Lyrics-Fanbase-Correlator\Lyric-Fanbase-Correlator\Processed_Reddit_Data\TaylorSwift_COMMENTS.csv"]
}

OUTPUT_DIRECTORY = "Processed_Artist_Data"


WINDOW_PRE = 14
WINDOW_POST = 14

# Artist Mapping for RAW files
FILENAME_MAP = {
    "playboicarti": "Playboi Carti",
    "SabrinaCarpenter": "Sabrina Carpenter",
    "TameImpala": "Tame Impala",
    "TaylorSwift": "Taylor Swift",
    "TheWeeknd": "The Weeknd",
    "billieeilish": "Billie Eilish",
    "Drizzy": "Drake",
    "Eminem": "Eminem",
    "greenday": "Green Day",
    "Jcole": "J. Cole",
    "JuiceWRLD": "Juice WRLD",
    "Kanye": "Kanye",
    "KendrickLamar": "Kendrick Lamar",
    "MacMiller": "Mac Miller"
}

# ALBUM RELEASE DATES (Updated with 2025 Releases)
ALBUM_DATES = {
    "Eminem": {
        "Recovery": "2010-06-18",
        "Music to be Murdered By": "2020-01-17",
        "The Death of Slim Shady": "2024-07-12"
    },
    "Taylor Swift": {
        "Speak Now": "2010-10-25", 
        "The Tortured Poets Department": "2024-04-19"
    },
    "Kanye": {
        "My Beautiful Dark Twisted Fantasy": "2010-11-22",
        "Vultures 1": "2024-02-10",
        "Vultures 2": "2024-08-03"
    },
    "Kendrick Lamar": {
        "good kid, m.A.A.d city": "2012-10-22",
        "Mr. Morale & the Big Steppers": "2022-05-13",
        "GNX": "2024-11-22"
    },
    "Drake": {
        "Views": "2016-04-29",
        "For All The Dogs": "2023-10-06",
        "Some Sexy Songs 4 U": "2025-02-14"
    },
    "The Weeknd": {
        "Kiss Land": "2013-09-10",
        "Dawn FM": "2022-01-07",
        "Hurry Up Tomorrow": "2025-01-31"
    },
    "J. Cole": {
        "Born Sinner": "2013-06-18",
        "The Off-Season": "2021-05-14",
        "Might Delete Later": "2024-04-05"
    },
    "Billie Eilish": {
        "WHEN WE ALL FALL ASLEEP": "2019-03-29",
        "Happier Than Ever": "2021-07-30",
        "HIT ME HARD AND SOFT": "2024-05-17"
    },
    "Playboi Carti": {
        "Die Lit": "2018-05-11",
        "Whole Lotta Red": "2020-12-25",
        "MUSIC": "2025-03-14"
    },
    "Juice WRLD": {
        "Goodbye & Good Riddance": "2018-05-23",
        "Fighting Demons": "2021-12-10",
        "The Party Never Ends": "2024-11-30"
    },
    "Mac Miller": {
        "Blue Slide Park": "2011-11-08",
        "Circles": "2020-01-17",
        "Balloonerism": "2025-01-17"
    },
    "Green Day": {
        "¡Uno!": "2012-09-21",
        "Father of All Motherfuckers": "2020-02-07",
        "Saviors": "2024-01-19"
    },
    "Tame Impala": {
        "Lonerism": "2012-10-05",
        "The Slow Rush": "2020-02-14",
        "Deadbeat": "2025-10-17"
    },
    "Sabrina Carpenter": {
        "Eyes Wide Open": "2015-04-14",
        "Short n' Sweet": "2024-08-23",
        "Man's Best Friend": "2021-06-04"
    }
}

BOT_PHRASES = ["i am a bot", "action was performed automatically", "submission has been removed", "contact the moderators", "message the mods"]
GIF_DOMAINS = ["giphy.com", "tenor.com", "imgur.com", ".gif"]

def get_artist_windows(artist_key):
    windows = []
    if artist_key not in ALBUM_DATES: return []
    for album, date_str in ALBUM_DATES[artist_key].items():
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
            start = dt - datetime.timedelta(days=WINDOW_PRE)
            end = dt + datetime.timedelta(days=WINDOW_POST)
            windows.append({"album": album, "start_ts": int(start.timestamp()), "end_ts": int(end.timestamp())})
        except ValueError: continue
    return windows

def check_window(created_utc, windows):
    for w in windows:
        if w["start_ts"] <= created_utc <= w["end_ts"]: return w["album"]
    return None

def clean_text(text):
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'http\S+', '', text)
    return " ".join(text.split())

def is_spam_or_bot(text):
    t_lower = text.lower()
    if t_lower in ["[removed]", "[deleted]", ""]: return True
    if any(d in t_lower for d in GIF_DOMAINS): return True
    if any(p in t_lower for p in BOT_PHRASES): return True
    return False

def process_raw_files(artist_name, file_prefix, windows, writer):
    # Process old JSONL files from the 'raw' folder
    target_files = [f"{file_prefix}_comments", f"{file_prefix}_submissions"]
    count = 0
    
    for fname in target_files:
        full_path = os.path.join(INPUT_DIRECTORY, fname)
        if not os.path.exists(full_path): continue
        
        print(f"    Reading RAW: {fname}...")
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                # Using tqdm here for large raw files
                for line in tqdm(f, desc=f"Scanning {fname}", unit="lines"):
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        created = int(obj.get('created_utc', 0))
                        
                        rel_album = check_window(created, windows)
                        if not rel_album: continue
                        
                        body = obj.get('body') or obj.get('selftext') or obj.get('title') or ""
                        if is_spam_or_bot(body): continue
                        
                        clean = clean_text(body)
                        if len(clean) < 3: continue
                        
                        r_date = datetime.datetime.fromtimestamp(created, tz=datetime.timezone.utc).strftime('%Y-%m-%d')
                        writer.writerow([obj.get('id'), artist_name, rel_album, r_date, clean, obj.get('score')])
                        count += 1
                    except: continue
        except Exception as e: print(f"    Error: {e}")
    return count

def process_csv_files(artist_name, windows, writer):
    # Process new 2025 CSV files
    if artist_name not in CSV_SOURCES: return 0
    
    count = 0
    for csv_path in CSV_SOURCES[artist_name]:
        if not os.path.exists(csv_path): 
            print(f"    Warning: CSV not found: {csv_path}")
            continue
        
        print(f"    Reading CSV: {os.path.basename(csv_path)}...")
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                csv_reader = csv.DictReader(f)
                
                for row in csv_reader:
                    try:
                        # Handle different timestamp formats
                        created = row.get('created_utc') or row.get('created') or row.get('timestamp')
                        if created:
                            created = int(float(created))
                        else:
                            continue

                        rel_album = check_window(created, windows)
                        if not rel_album: continue

                        body = row.get('body') or row.get('text') or row.get('selftext') or row.get('title') or ""
                        if is_spam_or_bot(body): continue
                        
                        clean = clean_text(body)
                        if len(clean) < 3: continue

                        r_date = datetime.datetime.fromtimestamp(created, tz=datetime.timezone.utc).strftime('%Y-%m-%d')
                        
                        row_id = row.get('id') or "csv_import"
                        score = row.get('score') or 0
                        
                        writer.writerow([row_id, artist_name, rel_album, r_date, clean, score])
                        count += 1
                    except: continue
        except Exception as e: print(f"    Error reading CSV: {e}")
    return count

def main():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    print(f"Starting Processing (Window: -{WINDOW_PRE} days / +{WINDOW_POST} days)...")
    
    for file_prefix, artist_name in FILENAME_MAP.items():
        output_filename = os.path.join(OUTPUT_DIRECTORY, f"{artist_name}_Filtered.csv")
        windows = get_artist_windows(artist_name)
        
        if not windows:
            print(f"Skipping {artist_name} (No Dates)")
            continue

        print(f"--> Processing {artist_name}...")
        
        with open(output_filename, 'w', newline='', encoding='utf-8') as out_file:
            writer = csv.writer(out_file)