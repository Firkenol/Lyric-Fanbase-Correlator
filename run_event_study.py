import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind

REDDIT_DIR = r"D:\Lyrics-Fanbase-Correlator\Final_Analysis_Results"
LYRICS_DIR = r"D:\Lyrics-Fanbase-Correlator\Lyrics_VAD_Data"
OUTPUT_DIR = r"D:\Lyrics-Fanbase-Correlator\Event_Study_Results"
GRAPH_DIR = r"D:\Lyrics-Fanbase-Correlator\Event_Study_Graphs"

ALBUM_DATES = {
    "Recovery": "2010-06-18", "Music to be Murdered By": "2020-01-17", "The Death of Slim Shady": "2024-07-12",
    "Speak Now": "2010-10-25", "The Tortured Poets Department": "2024-04-19", "The Life of a Showgirl": "2025-01-01", 
    "Eyes Wide Open": "2015-04-14", "Short n' Sweet": "2024-08-23", "Man's Best Friend": "2025-01-01", 
    "My Beautiful Dark Twisted Fantasy": "2010-11-22", "Vultures 1": "2024-02-10", "Vultures 2": "2024-08-03",
    "Views": "2016-04-29", "For All The Dogs": "2023-10-06", "Some Sexy Songs 4 U": "2025-01-01", 
    "good kid, m.A.A.d city": "2012-10-22", "Mr. Morale & the Big Steppers": "2022-05-13", "GNX": "2024-11-22",
    "Goodbye & Good Riddance": "2018-05-23", "Fighting Demons": "2021-12-10", "The Party Never Ends": "2025-01-01", 
    "WHEN WE ALL FALL ASLEEP, WHERE DO WE GO?": "2019-03-29", "Happier Than Ever": "2021-07-30", "HIT ME HARD AND SOFT": "2024-05-17",
    "Kiss Land": "2013-09-10", "Dawn FM": "2022-01-07", "Hurry Up Tomorrow": "2025-01-01", 
    "¡Uno!": "2012-09-21", "Father of All Motherfuckers": "2020-02-07", "Saviors": "2024-01-19",
    "Born Sinner": "2013-06-18", "The Off-Season": "2021-05-14", "Might Delete Later": "2024-04-05",
    "Blue Slide Park": "2011-11-08", "Circles": "2020-01-17", "Balloonerism": "2025-01-01", 
    "Die Lit": "2018-05-11", "Whole Lotta Red": "2020-12-25", "MUSIC": "2025-01-01", 
    "Lonerism": "2012-10-05", "The Slow Rush": "2020-02-14", "Deadbeat": "2025-01-01" 
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
        if not os.path.exists(d): 
            os.makedirs(d)

    album_date_df = pd.DataFrame(list(ALBUM_DATES.items()), columns=['Album', 'Release_Date'])
    album_date_df['Release_Date'] = pd.to_datetime(album_date_df['Release_Date'])

    reddit_files = [f for f in os.listdir(REDDIT_DIR) if f.endswith('_FullDist.csv')]
    master_stats = []

    for rf in reddit_files:
        artist = rf.replace('_FullDist.csv', '')
        lyrics_file = os.path.join(LYRICS_DIR, f"{artist}_Lyrics_VAD.csv")
        
        if not os.path.exists(lyrics_file): 
            continue
            
        print(f"Running Event Study for {artist}...")
        lyrics_df = pd.read_csv(lyrics_file)
        lyrics_df['Date'] = pd.to_datetime(lyrics_df['Date'])
        
        reddit_df = pd.read_csv(os.path.join(REDDIT_DIR, rf), low_memory=False)
        reddit_df['Date'] = pd.to_datetime(reddit_df['Date'], errors='coerce')
        reddit_df = reddit_df.dropna(subset=['Date'])
        reddit_df = calculate_vad(reddit_df)

        artist_albums = lyrics_df.merge(album_date_df, left_on='Date', right_on='Release_Date', how='inner')
        album_deltas = []

        for _, row in artist_albums.iterrows():
            release_date = row['Release_Date']
            album_name = row['Album']
            
            pre_mask = (reddit_df['Date'] >= release_date - pd.Timedelta(days=14)) & (reddit_df['Date'] < release_date)
            post_mask = (reddit_df['Date'] >= release_date) & (reddit_df['Date'] <= release_date + pd.Timedelta(days=14))
            
            pre_data = reddit_df[pre_mask]
            post_data = reddit_df[post_mask]
            
            if len(pre_data) < 5 or len(post_data) < 5:
                continue 
                
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

        if not album_deltas: 
            continue
            
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
        print("Event Study Complete. Check the Output and Graph folders.")
    else:
        print("No valid data found to correlate.")

if __name__ == "__main__":
    run_event_study()