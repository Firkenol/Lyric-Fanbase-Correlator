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

# load huggingface pipeline only if needed
classifier = None

def load_ai():
    global classifier
    if classifier is None:
        print("\n[SYSTEM] Loading GoEmotions AI Model into memory...")
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline(
            "text-classification", 
            model="SamLowe/roberta-base-go_emotions", 
            top_k=None, 
            device=device
        )
    return classifier

def process_missing_ai_data(df, file_path):
    # setup text col
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    text_col = next((cols_lower[c] for c in ['comment', 'body', 'selftext', 'title', 'text', 'content'] if c in cols_lower), None)
    
    if not text_col:
        print(f"   [!] Cannot run AI on {os.path.basename(file_path)}: No text column found.")
        return df

    print(f"   [AI] Processing {len(df)} rows for {os.path.basename(file_path)}. This will take a while...")
    clf = load_ai()
    
    df[text_col] = df[text_col].astype(str).fillna("")
    texts = df[text_col].tolist()
    
    # process in batches to prevent oom
    batch_size = 128
    all_scores = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Running Inference"):
        batch = texts[i:i+batch_size]
        # truncate strings to fit roberta limit
        batch = [str(t)[:512] for t in batch]
        results = clf(batch)
        
        for res in results:
            score_dict = {pred['label']: pred['score'] for pred in res}
            all_scores.append(score_dict)
            
    scores_df = pd.DataFrame(all_scores)
    df = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
    
    # save so we don't have to do this again
    print(f"   [AI] Saving processed data back to {os.path.basename(file_path)}...")
    df.to_csv(file_path, index=False)
    return df

def get_artist_from_filename(filename):
    f = filename.lower().replace(" ", "")
    for sub, artist in SUBREDDIT_MAP.items():
        if sub in f or artist.lower().replace(" ", "") in f:
            return artist
    return None

def calculate_vad(df):
    emotions = [e for e in VAD_DICT.keys() if e in df.columns]
    if not emotions:
        return df
        
    v_weights = pd.Series({e: VAD_DICT[e][0] for e in emotions})
    a_weights = pd.Series({e: VAD_DICT[e][1] for e in emotions})
    d_weights = pd.Series({e: VAD_DICT[e][2] for e in emotions})
    
    prob_sum = df[emotions].sum(axis=1).replace(0, 1)
    
    df['Valence'] = df[emotions].dot(v_weights) / prob_sum
    df['Arousal'] = df[emotions].dot(a_weights) / prob_sum
    df['Dominance'] = df[emotions].dot(d_weights) / prob_sum
    return df

def run_event_study():
    for d in [OUTPUT_DIR, GRAPH_DIR]:
        if not os.path.exists(d): os.makedirs(d)

    raw_lyrics = pd.read_csv(LYRICS_FILE)
    raw_lyrics['clean_artist'] = raw_lyrics['Artist'].astype(str).str.lower().str.replace(" ", "")

    all_files = []
    if os.path.exists(DIR_ANALYSIS):
        all_files.extend([os.path.join(DIR_ANALYSIS, f) for f in os.listdir(DIR_ANALYSIS) if f.endswith('.csv')])
    if os.path.exists(DIR_PROCESSED):
        all_files.extend([os.path.join(DIR_PROCESSED, f) for f in os.listdir(DIR_PROCESSED) if f.endswith('.csv')])

    artist_files = {}
    for path in all_files:
        filename = os.path.basename(path)
        artist = get_artist_from_filename(filename)
        if artist:
            if artist not in artist_files: artist_files[artist] = []
            artist_files[artist].append(path)

    master_stats = []
    WINDOW_DAYS = 14
    MIN_POSTS = 3

    for artist, files in artist_files.items():
        artist_songs = raw_lyrics[raw_lyrics['clean_artist'] == artist.lower().replace(" ", "")].copy()
        if artist_songs.empty: continue
            
        print(f"\n--- Running Event Study for {artist} ---")
        
        album_vad = artist_songs.groupby('Album')[['Valence', 'Arousal', 'Dominance']].mean().reset_index()
        album_vad['Release_Date'] = album_vad['Album'].map(ALBUM_DATES)
        album_vad['Release_Date'] = pd.to_datetime(album_vad['Release_Date'])
        album_vad = album_vad.dropna(subset=['Release_Date'])
        
        merged_reddit = []
        
        for f in files:
            df = pd.read_csv(f, low_memory=False, on_bad_lines='skip')
            
            # ghost town check
            if len(df) == 0:
                continue
            
            # trigger ai scoring ONLY for the 2025/missing files
            if 'joy' not in df.columns or 'anger' not in df.columns:
                if "MASTER" in f.upper() or "COMMENTS" in f.upper():
                    df = process_missing_ai_data(df, f)
                else:
                    # Ignore the raw duplicates like Eminem_Filtered.csv
                    continue
            
            # parse dates dynamically
            cols_lower = {str(c).lower().strip(): c for c in df.columns}
            date_col = next((cols_lower[c] for c in ['date', 'created_utc', 'timestamp', 'utc'] if c in cols_lower), None)
            
            if not date_col: continue
                
            df['Parsed_Date'] = pd.to_datetime(df[date_col], errors='coerce', unit='s' if df[date_col].dtype != 'object' else None)
            df = df.dropna(subset=['Parsed_Date'])
            
            if 'joy' in df.columns and 'anger' in df.columns:
                merged_reddit.append(df)
                
        if not merged_reddit:
            print(f"   [!] No valid data found for {artist}.")
            continue
            
        reddit_df = pd.concat(merged_reddit)
        reddit_df = calculate_vad(reddit_df)

        album_deltas = []

        for _, row in album_vad.iterrows():
            release_date = row['Release_Date']
            album_name = row['Album']
            
            pre_mask = (reddit_df['Parsed_Date'] >= release_date - pd.Timedelta(days=WINDOW_DAYS)) & (reddit_df['Parsed_Date'] < release_date)
            post_mask = (reddit_df['Parsed_Date'] >= release_date) & (reddit_df['Parsed_Date'] <= release_date + pd.Timedelta(days=WINDOW_DAYS))
            
            pre_data = reddit_df[pre_mask]
            post_data = reddit_df[post_mask]
            
            if len(pre_data) < MIN_POSTS or len(post_data) < MIN_POSTS:
                print(f"   [!] Dropped '{album_name}': Pre-posts: {len(pre_data)}, Post-posts: {len(post_data)}")
                continue 
                
            print(f"   [OK] Processed '{album_name}'")
            
            album_record = {'Artist': artist, 'Album': album_name}
            for dim in ['Valence', 'Arousal', 'Dominance']:
                pre_mean = pre_data[dim].mean()
                post_mean = post_data[dim].mean()
                delta = post_mean - pre_mean
                
                t_stat, p_val = ttest_ind(pre_data[dim], post_data[dim], equal_var=False)
                
                album_record[f'Pre_{dim}'] = pre_mean
                album_record[f'Post_{dim}'] = post_mean
                album_record[f'Delta_{dim}'] = delta
                album_record[f'TTest_p_{dim}'] = p_val
                album_record[f'Lyric_{dim}'] = row[dim]
                
            album_deltas.append(album_record)

        if not album_deltas: continue
            
        delta_df = pd.DataFrame(album_deltas)
        delta_df.to_csv(os.path.join(OUTPUT_DIR, f"{artist}_Album_Shifts.csv"), index=False)
        
        artist_results = {'Artist': artist, 'Valid_Albums': len(delta_df)}
        
        for dim in ['Valence', 'Arousal', 'Dominance']:
            if len(delta_df) > 2:
                r_val, p_val = pearsonr(delta_df[f'Lyric_{dim}'], delta_df[f'Delta_{dim}'])
                artist_results[f'{dim}_Pearson_r'] = r_val
                artist_results[f'{dim}_Pearson_p'] = p_val
                
                plt.figure(figsize=(8, 6))
                sns.regplot(x=delta_df[f'Lyric_{dim}'], y=delta_df[f'Delta_{dim}'], scatter_kws={'s':100}, line_kws={'color':'red'})
                
                for i, txt in enumerate(delta_df['Album']):
                    plt.annotate(txt, (delta_df[f'Lyric_{dim}'][i], delta_df[f'Delta_{dim}'][i]), xytext=(5,5), textcoords='offset points', fontsize=8)
                    
                plt.title(f"{artist}: Does {dim} in Music Predict Fanbase Shift?")
                plt.xlabel(f"Music Lyrics {dim} (-1 to 1)")
                plt.ylabel(f"Fanbase Reddit Shift (Post - Pre)")
                plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
                plt.savefig(os.path.join(GRAPH_DIR, f"{artist}_{dim}_Scatter.png"))
                plt.close()
                
        master_stats.append(artist_results)

    if master_stats:
        final_df = pd.DataFrame(master_stats)
        final_df.to_csv(os.path.join(OUTPUT_DIR, "Master_Correlation_Results.csv"), index=False)
        print("\nAll Done! Master Correlation Table Generated.")

if __name__ == "__main__":
    run_event_study()