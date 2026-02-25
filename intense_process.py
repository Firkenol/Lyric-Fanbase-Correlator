import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
import torch
from transformers import pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DIR_ANALYSIS = r"D:\Lyrics-Fanbase-Correlator\Final_Analysis_Results"
DIR_PROCESSED = r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"
# We will overwrite this file with the new, unified lyrics data
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

# --- GLOBAL MODEL LOADER ---
classifier = None

def load_ai():
    global classifier
    if classifier is None:
        print("\n[SYSTEM] Loading GoEmotions AI Model...")
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=device)
    return classifier

# --- STEP 1: DYNAMIC LYRIC PROCESSING ---
def update_lyric_scores():
    print("\n--- STEP 1: Refreshing Lyric VAD Scores ---")
    raw_lyrics = pd.read_csv(LYRICS_FILE)
    
    # ensure columns exist
    if 'joy' not in raw_lyrics.columns:
        print("   [!] Lyrics file missing AI scores. Running RoBERTa on lyrics...")
        clf = load_ai()
        lyrics_text = raw_lyrics['Lyric'].astype(str).tolist()
        results = []
        for i in tqdm(range(0, len(lyrics_text), 32), desc="Scoring Lyrics"):
            batch = [s[:512] for s in lyrics_text[i:i+32]]
            results.extend(clf(batch))
        
        scores = []
        for res in results:
            scores.append({pred['label']: pred['score'] for pred in res})
        scores_df = pd.DataFrame(scores)
        raw_lyrics = pd.concat([raw_lyrics, scores_df], axis=1)

    # calculate vad
    emotions = [e for e in VAD_DICT.keys() if e in raw_lyrics.columns]
    v_w = pd.Series({e: VAD_DICT[e][0] for e in emotions})
    a_w = pd.Series({e: VAD_DICT[e][1] for e in emotions})
    d_w = pd.Series({e: VAD_DICT[e][2] for e in emotions})
    
    prob_sum = raw_lyrics[emotions].sum(axis=1).replace(0, 1)
    raw_lyrics['Valence'] = raw_lyrics[emotions].dot(v_w) / prob_sum
    raw_lyrics['Arousal'] = raw_lyrics[emotions].dot(a_w) / prob_sum
    raw_lyrics['Dominance'] = raw_lyrics[emotions].dot(d_w) / prob_sum
    
    raw_lyrics.to_csv(LYRICS_FILE, index=False)
    print(f"   [OK] Lyrics VAD updated. Saved to {LYRICS_FILE}")
    return raw_lyrics

# --- STEP 2: REDDIT AI SCORING ---
def score_reddit_file(df, file_path):
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    text_col = next((cols_lower[c] for c in ['comment', 'body', 'selftext', 'text'] if c in cols_lower), None)
    
    if not text_col:
        return df

    print(f"   [AI] Scoring {len(df)} posts in {os.path.basename(file_path)}...")
    clf = load_ai()
    texts = df[text_col].astype(str).tolist()
    
    all_scores = []
    batch_size = 128
    for i in tqdm(range(0, len(texts), batch_size), desc="Reddit Inference"):
        batch = [t[:512] for t in texts[i:i+batch_size]]
        results = clf(batch)
        for res in results:
            all_scores.append({pred['label']: pred['score'] for pred in res})
            
    scores_df = pd.DataFrame(all_scores)
    df = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
    df.to_csv(file_path, index=False)
    return df

# --- STEP 3: THE MAIN LOOP ---
def run_everything():
    for d in [OUTPUT_DIR, GRAPH_DIR]:
        if not os.path.exists(d): os.makedirs(d)

    # get fresh lyrics
    lyrics_df = update_lyric_scores()
    lyrics_df['clean_artist'] = lyrics_df['Artist'].astype(str).str.lower().str.replace(" ", "")

    # gather files
    all_paths = []
    for folder in [DIR_ANALYSIS, DIR_PROCESSED]:
        if os.path.exists(folder):
            all_paths.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')])

    artist_files = {}
    for p in all_paths:
        fname = os.path.basename(p).lower().replace(" ", "")
        for sub, artist in SUBREDDIT_MAP.items():
            if sub in fname or artist.lower().replace(" ", "") in fname:
                if artist not in artist_files: artist_files[artist] = []
                artist_files[artist].append(p)

    master_stats = []
    
    for artist, files in artist_files.items():
        print(f"\n--- Processing {artist} ---")
        
        # calculate album means from lyrics
        artist_songs = lyrics_df[lyrics_df['clean_artist'] == artist.lower().replace(" ", "")].copy()
        album_vad = artist_songs.groupby('Album')[['Valence', 'Arousal', 'Dominance']].mean().reset_index()
        
        # bulletproof date mapping
        lower_dates = {k.lower().strip(): v for k, v in ALBUM_DATES.items()}
        album_vad['Release_Date'] = album_vad['Album'].str.lower().str.strip().map(lower_dates)
        album_vad = album_vad.dropna(subset=['Release_Date'])
        album_vad['Release_Date'] = pd.to_datetime(album_vad['Release_Date'])

        if album_vad.empty:
            print(f"   [!] No release dates found for {artist}'s albums. Skipping.")
            continue

        # load and score reddit data
        merged_reddit = []
        for f in files:
            # skip duplicate raw files
            if f.lower().endswith("filtered.csv") and any("fulldist" in x.lower() for x in files):
                continue
                
            df = pd.read_csv(f, low_memory=False, on_bad_lines='skip')
            if len(df) == 0: continue
            
            if 'joy' not in df.columns:
                df = score_reddit_file(df, f)
            
            # calculate vad for reddit
            emotions = [e for e in VAD_DICT.keys() if e in df.columns]
            v_w = pd.Series({e: VAD_DICT[e][0] for e in emotions})
            a_w = pd.Series({e: VAD_DICT[e][1] for e in emotions})
            d_w = pd.Series({e: VAD_DICT[e][2] for e in emotions})
            p_sum = df[emotions].sum(axis=1).replace(0, 1)
            df['Valence'] = df[emotions].dot(v_w) / p_sum
            df['Arousal'] = df[emotions].dot(a_w) / p_sum
            df['Dominance'] = df[emotions].dot(d_w) / p_sum
            
            # date parsing
            cols = {str(c).lower(): c for c in df.columns}
            d_col = next((cols[c] for c in ['date', 'created_utc', 'timestamp'] if c in cols), None)
            if d_col:
                df['Parsed_Date'] = pd.to_datetime(df[d_col], errors='coerce', unit='s' if df[d_col].dtype != 'object' else None)
                merged_reddit.append(df.dropna(subset=['Parsed_Date']))

        if not merged_reddit: continue
        full_reddit = pd.concat(merged_reddit)

        # event study
        deltas = []
        for _, row in album_vad.iterrows():
            rel = row['Release_Date']
            pre = full_reddit[(full_reddit['Parsed_Date'] >= rel - pd.Timedelta(days=14)) & (full_reddit['Parsed_Date'] < rel)]
            post = full_reddit[(full_reddit['Parsed_Date'] >= rel) & (full_reddit['Parsed_Date'] <= rel + pd.Timedelta(days=14))]
            
            if len(pre) < 3 or len(post) < 3:
                print(f"   [!] Dropped '{row['Album']}': Insufficient posts.")
                continue
            
            print(f"   [OK] Processed '{row['Album']}'")
            rec = {'Artist': artist, 'Album': row['Album']}
            for dim in ['Valence', 'Arousal', 'Dominance']:
                rec[f'Delta_{dim}'] = post[dim].mean() - pre[dim].mean()
                rec[f'Lyric_{dim}'] = row[dim]
            deltas.append(rec)

        if deltas:
            df_deltas = pd.DataFrame(deltas)
            df_deltas.to_csv(os.path.join(OUTPUT_DIR, f"{artist}_Shifts.csv"), index=False)
            
            summary = {'Artist': artist, 'Albums': len(df_deltas)}
            for dim in ['Valence', 'Arousal', 'Dominance']:
                if len(df_deltas) >= 2:
                    r, p = pearsonr(df_deltas[f'Lyric_{dim}'], df_deltas[f'Delta_{dim}'])
                    summary[f'{dim}_r'] = r
                    summary[f'{dim}_p'] = p
                    
                    # graph it
                    plt.figure()
                    sns.regplot(data=df_deltas, x=f'Lyric_{dim}', y=f'Delta_{dim}')
                    plt.title(f"{artist} {dim} Correlation")
                    plt.savefig(os.path.join(GRAPH_DIR, f"{artist}_{dim}.png"))
                    plt.close()
            master_stats.append(summary)

    pd.DataFrame(master_stats).to_csv(os.path.join(OUTPUT_DIR, "Final_Master_Correlations.csv"), index=False)
    print("\n--- DONE. Go to sleep. ---")

if __name__ == "__main__":
    run_everything()