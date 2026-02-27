import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
import torch
from transformers import pipeline
from tqdm import tqdm
import lyricsgenius
from dotenv import load_dotenv
import warnings
import re

warnings.filterwarnings('ignore')

# init env
load_dotenv()
GENIUS_TOKEN = "t7Dyu5aBr6n5ylo0napiP4N7N5sIli5NHBwEC79ratXrKs_QdQzGINPczcEYkjsm"

if not GENIUS_TOKEN:
    print("Error: GENIUS_ACCESS_TOKEN not found in .env file.")
    sys.exit()

DIR_ANALYSIS = r"D:\Lyrics-Fanbase-Correlator\Final_Analysis_Results"
DIR_PROCESSED = r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"
LYRICS_FILE = r"D:\Lyrics-Fanbase-Correlator\song_level_roberta_vad_fixed.csv"
OUTPUT_DIR = r"D:\Lyrics-Fanbase-Correlator\Event_Study_Results"
GRAPH_DIR = r"D:\Lyrics-Fanbase-Correlator\Event_Study_Graphs"

ALBUM_DATES = {
    "Recovery": "2010-06-18", "Music to be Murdered By": "2020-01-17", "The Death of Slim Shady": "2024-07-12",
    "Speak Now": "2010-10-25", "The Tortured Poets Department": "2024-04-19", "The Life of a Showgirl": "2025-10-03", 
    "Eyes Wide Open": "2015-04-14", "Short n' Sweet": "2024-08-23", "Man's Best Friend": "2025-08-29", 
    "The Life of Pablo": "2016-02-14", "Vultures 1": "2024-02-10", "Vultures 2": "2024-08-03",
    "Views": "2016-04-29", "For All The Dogs": "2023-10-06", "Some Sexy Songs 4 U": "2025-02-14", 
    "good kid, m.A.A.d city": "2012-10-22", "Mr. Morale & the Big Steppers": "2022-05-13", "GNX": "2024-11-22",
    "Goodbye & Good Riddance": "2018-05-23", "Fighting Demons": "2021-12-10", "The Party Never Ends": "2024-11-29", 
    "WHEN WE ALL FALL ASLEEP, WHERE DO WE GO?": "2019-03-29", "Happier Than Ever": "2021-07-30", "HIT ME HARD AND SOFT": "2024-05-17",
    "Kiss Land": "2013-09-10", "Dawn FM": "2022-01-07", "Hurry Up Tomorrow": "2025-01-31", 
    "¡Uno!": "2012-09-21", "Father of All Motherfuckers": "2020-02-07", "Saviors": "2024-01-19",
    "Born Sinner": "2013-06-18", "The Off-Season": "2021-05-14", "Might Delete Later": "2024-04-05",
    "Swimming": "2018-08-03", "Circles": "2020-01-17", "Balloonerism": "2025-01-17", 
    "Die Lit": "2018-05-11", "Whole Lotta Red": "2020-12-25", "MUSIC": "2025-03-14", 
    "Lonerism": "2012-10-05", "The Slow Rush": "2020-02-14", "Deadbeat": "2025-10-17" 
}

SUBREDDIT_MAP = {
    "eminem": "Eminem", "taylorswift": "Taylor Swift", "sabrinacarpenter": "Sabrina Carpenter",
    "kanye": "Kanye West", "drizzy": "Drake", "kendricklamar": "Kendrick Lamar",
    "juicewrld": "Juice WRLD", "billieeilish": "Billie Eilish", "theweeknd": "The Weeknd",
    "greenday": "Green Day", "jcole": "J. Cole", "macmiller": "Mac Miller",
    "playboicarti": "Playboi Carti", "tameimpala": "Tame Impala", "drake": "Drake"
}

VAD_DICT = {
    'admiration': [0.88, 0.52, 0.71], 'amusement': [0.89, 0.65, 0.58], 'anger': [0.16, 0.86, 0.65],
    'annoyance': [0.22, 0.68, 0.55], 'approval': [0.77, 0.48, 0.66], 'caring': [0.82, 0.45, 0.51],
    'confusion': [0.35, 0.55, 0.30], 'curiosity': [0.65, 0.60, 0.50], 'desire': [0.76, 0.70, 0.60],
    'disappointment': [0.23, 0.45, 0.35], 'disapproval': [0.25, 0.55, 0.58], 'disgust': [0.10, 0.68, 0.50],
    'embarrassment': [0.20, 0.60, 0.25], 'excitement': [0.85, 0.82, 0.56], 'fear': [0.07, 0.82, 0.21],
    'gratitude': [0.87, 0.55, 0.52], 'grief': [0.05, 0.54, 0.21], 'joy': [0.96, 0.61, 0.75],
    'love': [0.98, 0.65, 0.55], 'nervousness': [0.25, 0.75, 0.30], 'optimism': [0.80, 0.55, 0.65],
    'pride': [0.85, 0.60, 0.75], 'realization': [0.60, 0.55, 0.50], 'relief': [0.75, 0.35, 0.55],
    'remorse': [0.15, 0.45, 0.25], 'sadness': [0.08, 0.34, 0.22], 'surprise': [0.65, 0.85, 0.40],
    'neutral': [0.50, 0.50, 0.50]
}

classifier = None

def load_ai():
    global classifier
    if classifier is None:
        print("\n[SYSTEM] Loading GoEmotions AI Model...")
        device = 0 if torch.cuda.is_available() else -1
        # strict token limit enforced here
        classifier = pipeline(
            "text-classification", 
            model="SamLowe/roberta-base-go_emotions", 
            top_k=None, 
            device=device,
            truncation=True,
            max_length=512
        )
    return classifier

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # drop urls
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # drop weird unicode
    text = re.sub(r'[^A-Za-z0-9\s.,!?\']', '', text)
    # squash spaces
    return re.sub(r'\s+', ' ', text).strip()

def get_lyrics_from_genius(artist_name, album_name):
    print(f"   [GENIUS] Fetching lyrics for {artist_name} - {album_name}...")
    genius = lyricsgenius.Genius(GENIUS_TOKEN, verbose=False, remove_section_headers=True)
    try:
        album = genius.search_album(album_name, artist_name)
        if album:
            return [t.song.lyrics for t in album.tracks]
    except Exception as e:
        print(f"   [!] Genius Error for {album_name}: {e}")
    return None

def calculate_vad_for_df(df):
    emotions = [e for e in VAD_DICT.keys() if e in df.columns]
    vw = pd.Series({e: VAD_DICT[e][0] for e in emotions})
    aw = pd.Series({e: VAD_DICT[e][1] for e in emotions})
    dw = pd.Series({e: VAD_DICT[e][2] for e in emotions})
    psum = df[emotions].sum(axis=1).replace(0, 1)
    df['Valence'] = df[emotions].dot(vw) / psum
    df['Arousal'] = df[emotions].dot(aw) / psum
    df['Dominance'] = df[emotions].dot(dw) / psum
    return df

def run_everything():
    for d in [OUTPUT_DIR, GRAPH_DIR]:
        if not os.path.exists(d): os.makedirs(d)

    lyrics_df = pd.read_csv(LYRICS_FILE)
    
    found_albums_clean = set(lyrics_df['Album'].str.lower().str.replace(" ", "").unique())
    new_rows = []
    
    for album, date in ALBUM_DATES.items():
        clean_alb = album.lower().replace(" ", "")
        if clean_alb not in found_albums_clean:
            artist_name = "Unknown"
            for sub, art in SUBREDDIT_MAP.items():
                if art.lower().replace(" ", "") in clean_alb or any(art.lower() in x.lower() for x in [album, artist_name]):
                    artist_name = art
            
            # hardcode tlop mapping
            if "pablo" in clean_alb: artist_name = "Kanye West"
            
            track_lyrics = get_lyrics_from_genius(artist_name, album)
            if track_lyrics:
                clf = load_ai()
                print(f"   [AI] Scoring {len(track_lyrics)} tracks for {album}...")
                for lyr in tqdm(track_lyrics, desc=f"Scoring {album}"):
                    res = clf(str(lyr)[:2000], truncation=True)[0]
                    row_dict = {pred['label']: pred['score'] for pred in res}
                    row_dict.update({'Artist': artist_name, 'Album': album})
                    new_rows.append(row_dict)

    if new_rows:
        lyrics_df = pd.concat([lyrics_df, pd.DataFrame(new_rows)], ignore_index=True)
        lyrics_df = calculate_vad_for_df(lyrics_df)
        lyrics_df.to_csv(LYRICS_FILE, index=False)
        print("   [OK] Lyrics CSV updated.")

    lyrics_df['clean_artist'] = lyrics_df['Artist'].astype(str).str.lower().str.replace(" ", "")

    all_paths = [os.path.join(fld, f) for fld in [DIR_ANALYSIS, DIR_PROCESSED] if os.path.exists(fld) for f in os.listdir(fld) if f.endswith('.csv')]
    artist_files = {}
    for p in all_paths:
        fn = os.path.basename(p).lower().replace(" ", "")
        for sub, art in SUBREDDIT_MAP.items():
            if sub in fn or art.lower().replace(" ", "") in fn:
                if art not in artist_files: artist_files[art] = []
                artist_files[art].append(p)

    master_stats = []
    for art, files in artist_files.items():
        print(f"\n--- Processing {art} ---")
        art_songs = lyrics_df[lyrics_df['clean_artist'] == art.lower().replace(" ", "")].copy()
        alb_vad = art_songs.groupby('Album')[['Valence', 'Arousal', 'Dominance']].mean().reset_index()
        
        low_dates = {k.lower().strip(): v for k, v in ALBUM_DATES.items()}
        alb_vad['Release_Date'] = pd.to_datetime(alb_vad['Album'].str.lower().str.strip().map(low_dates))
        alb_vad = alb_vad.dropna(subset=['Release_Date'])

        merged_reddit = []
        for f in files:
            if f.lower().endswith("filtered.csv") and any("fulldist" in x.lower() for x in files): continue
            df = pd.read_csv(f, low_memory=False, on_bad_lines='skip')
            if len(df) == 0: continue
            
            if 'joy' not in df.columns:
                print(f"   [AI] Scoring Reddit Data: {os.path.basename(f)}")
                clf = load_ai()
                t_col = next((c for c in df.columns if c.lower() in ['comment', 'body', 'text']), None)
                if t_col:
                    # scrub text and drop empty rows
                    df[t_col] = df[t_col].apply(clean_text)
                    df = df[df[t_col].str.len() > 0]
                    
                    txts = df[t_col].astype(str).tolist()
                    scores = []
                    for i in tqdm(range(0, len(txts), 128), desc="Inference"):
                        # force truncation
                        res = clf([str(t)[:2000] for t in txts[i:i+128]], truncation=True)
                        scores.extend([{p['label']: p['score'] for p in r} for r in res])
                    
                    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores).reset_index(drop=True)], axis=1)
                    df.to_csv(f, index=False)

            df = calculate_vad_for_df(df)
            c_low = {str(c).lower(): c for c in df.columns}
            d_col = next((c_low[c] for c in ['date', 'created_utc', 'timestamp'] if c in c_low), None)
            if d_col:
                df['Parsed_Date'] = pd.to_datetime(df[d_col], errors='coerce', unit='s' if df[d_col].dtype != 'object' else None)
                merged_reddit.append(df.dropna(subset=['Parsed_Date']))

        if not merged_reddit: continue
        full_red = pd.concat(merged_reddit)
        
        deltas = []
        for _, row in alb_vad.iterrows():
            rel = row['Release_Date']
            pre = full_red[(full_red['Parsed_Date'] >= rel - pd.Timedelta(days=14)) & (full_red['Parsed_Date'] < rel)]
            post = full_red[(full_red['Parsed_Date'] >= rel) & (full_red['Parsed_Date'] <= rel + pd.Timedelta(days=14))]
            
            if len(pre) >= 3 and len(post) >= 3:
                print(f"   [OK] {row['Album']}")
                rec = {'Artist': art, 'Album': row['Album']}
                for dim in ['Valence', 'Arousal', 'Dominance']:
                    rec[f'Delta_{dim}'] = post[dim].mean() - pre[dim].mean()
                    rec[f'Lyric_{dim}'] = row[dim]
                deltas.append(rec)

        if deltas:
            df_d = pd.DataFrame(deltas)
            summary = {'Artist': art, 'Albums': len(df_d)}
            for dim in ['Valence', 'Arousal', 'Dominance']:
                if len(df_d) >= 2:
                    r, p = pearsonr(df_d[f'Lyric_{dim}'], df_d[f'Delta_{dim}'])
                    summary[f'{dim}_r'], summary[f'{dim}_p'] = r, p
                    plt.figure()
                    sns.regplot(data=df_d, x=f'Lyric_{dim}', y=f'Delta_{dim}')
                    plt.title(f"{art} {dim}")
                    plt.savefig(os.path.join(GRAPH_DIR, f"{art}_{dim}.png"))
                    plt.close()
            master_stats.append(summary)

    pd.DataFrame(master_stats).to_csv(os.path.join(OUTPUT_DIR, "Final_Results.csv"), index=False)
    print("\n--- ALL FINISHED. GRAPHS GENERATED. ---")

if __name__ == "__main__":
    run_everything()