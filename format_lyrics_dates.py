import pandas as pd
import os

INPUT_FILE = r"D:\Lyrics-Fanbase-Correlator\song_level_goemotions.csv"
OUTPUT_DIR = r"D:\Lyrics-Fanbase-Correlator\Lyrics_VAD_Data"

ALBUM_DATES = {
    "Recovery": "2010-06-18", "Music to be Murdered By": "2020-01-17", "The Death of Slim Shady": "2024-07-12",
    "Speak Now": "2010-10-25", "The Tortured Poets Department": "2024-04-19", "The Life of a Showgirl": "2025-10-03", 
    "Eyes Wide Open": "2015-04-14", "Short n' Sweet": "2024-08-23", "Man's Best Friend": "2025-08-29", 
    "My Beautiful Dark Twisted Fantasy": "2010-11-22", "Vultures 1": "2024-02-10", "Vultures 2": "2024-08-03",
    "Views": "2016-04-29", "For All The Dogs": "2023-10-06", "Some Sexy Songs 4 U": "2025-02-14", 
    "good kid, m.A.A.d city": "2012-10-22", "Mr. Morale & the Big Steppers": "2022-05-13", "GNX": "2024-11-22",
    "Goodbye & Good Riddance": "2018-05-23", "Fighting Demons": "2021-12-10", "The Party Never Ends": "2024-11-29", 
    "WHEN WE ALL FALL ASLEEP, WHERE DO WE GO?": "2019-03-29", "Happier Than Ever": "2021-07-30", "HIT ME HARD AND SOFT": "2024-05-17",
    "Kiss Land": "2013-09-10", "Dawn FM": "2022-01-07", "Hurry Up Tomorrow": "2025-01-31", 
    "¡Uno!": "2012-09-21", "Father of All Motherfuckers": "2020-02-07", "Saviors": "2024-01-19",
    "Born Sinner": "2013-06-18", "The Off-Season": "2021-05-14", "Might Delete Later": "2024-04-05",
    "Blue Slide Park": "2011-11-08", "Circles": "2020-01-17", "Balloonerism": "2025-01-17", 
    "Die Lit": "2018-05-11", "Whole Lotta Red": "2020-12-25", "MUSIC": "2025-03-14", 
    "Lonerism": "2012-10-05", "The Slow Rush": "2020-02-14", "Deadbeat": "2025-10-17" 
}

def prep_lyrics():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = pd.read_csv(INPUT_FILE)
    
    df['Date'] = df['Album'].map(ALBUM_DATES)
    df = df.dropna(subset=['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    artists = df['Artist'].unique()
    
    for artist in artists:
        artist_df = df[df['Artist'] == artist].copy()
        album_vad = artist_df.groupby('Date')[['Valence', 'Arousal', 'Dominance']].mean().reset_index()
        
        out_path = os.path.join(OUTPUT_DIR, f"{artist.replace(' ', '')}_Lyrics_VAD.csv")
        album_vad.to_csv(out_path, index=False)
        print(f"Prepared lyrics timeline for {artist}")

if __name__ == "__main__":
    prep_lyrics()