import os
import csv
import sys
import lyricsgenius
import statistics
import time
from dotenv import load_dotenv
from transformers import pipeline

# Load my environment variables
load_dotenv()
token = os.getenv("GENIUS_ACCESS_TOKEN")

if not token:
    print("Error: GENIUS_ACCESS_TOKEN not found.")
    sys.exit()

# Initialize Genius with robust timeout settings
genius = lyricsgenius.Genius(token, timeout=25, retries=3)
genius.verbose = False 
genius.remove_section_headers = True
genius.skip_non_songs = True

# --- NEW FILE NAMES ---
# I am naming these differently so I can compare them with the previous results
SONG_FILE = "song_level_goemotions.csv"
ALBUM_FILE = "album_level_goemotions.csv"

print("Loading GoEmotions BERT model...")
# Using the 28-emotion model instead of the basic 6-emotion one
classifier = pipeline("text-classification", model="monologg/bert-base-cased-goemotions-original", return_all_scores=True)

# --- 28-DIMENSION EMOTION MAP ---
# Mapping these specific 28 emotions to the VAD (Valence, Arousal, Dominance) scale
# Values are approx. -1.0 to +1.0
vad_map = {
    # High Energy / Positive
    'admiration':     [ 0.8,  0.4, -0.2], 
    'amusement':      [ 0.7,  0.5,  0.5],
    'excitement':     [ 0.9,  0.9,  0.7],
    'joy':            [ 0.95, 0.7,  0.8],
    'love':           [ 0.9,  0.6,  0.6],
    'pride':          [ 0.8,  0.7,  0.9], 
    'optimism':       [ 0.7,  0.5,  0.7],
    'gratitude':      [ 0.8,  0.3, -0.1],

    # High Energy / Negative
    'anger':          [-0.7,  0.9,  0.8], 
    'annoyance':      [-0.5,  0.6,  0.4],
    'disapproval':    [-0.6,  0.5,  0.6],
    'fear':           [-0.7,  0.9, -0.6], 
    'nervousness':    [-0.5,  0.8, -0.7],

    # Low Energy / Negative (Complex Sadness)
    'remorse':        [-0.8,  0.2, -0.5], # Regret/Guilt
    'sadness':        [-0.9, -0.2, -0.6],
    'disappointment': [-0.7,  0.1, -0.4],
    'embarrassment':  [-0.6,  0.5, -0.7],
    'grief':          [-0.9,  0.2, -0.5],

    # Cognitive / Neutral
    'confusion':      [-0.3,  0.6, -0.4],
    'curiosity':      [ 0.4,  0.6,  0.1],
    'realization':    [ 0.2,  0.5,  0.3],
    'surprise':       [ 0.5,  0.9,  0.1],
    'neutral':        [ 0.0,  0.0,  0.0],
    
    # Ambiguous
    'caring':         [ 0.7,  0.2,  0.4],
    'desire':         [ 0.6,  0.8,  0.4], 
    'relief':         [ 0.6, -0.3,  0.2],
    'approval':       [ 0.6,  0.3,  0.5],
}

def analyze_lyrics(text):
    chunk_size = 512
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    total_v, total_a, total_d = 0, 0, 0
    emotion_scores = {} 
    valid_chunks = 0
    
    for chunk in chunks:
        if len(chunk.strip()) < 10: continue
        
        try:
            results = classifier(chunk)[0]
            
            chunk_v, chunk_a, chunk_d = 0, 0, 0
            
            # Iterate through all 28 emotions for this chunk
            for res in results:
                lbl = res['label']
                prob = res['score']
                
                # Accumulate the score for determining the dominant emotion later
                emotion_scores[lbl] = emotion_scores.get(lbl, 0) + prob
                
                # Weighted VAD calculation
                if lbl in vad_map:
                    v, a, d = vad_map[lbl]
                    chunk_v += v * prob
                    chunk_a += a * prob
                    chunk_d += d * prob
            
            total_v += chunk_v
            total_a += chunk_a
            total_d += chunk_d
            valid_chunks += 1
            
        except Exception:
            continue
            
    if valid_chunks == 0:
        return 0.0, 0.0, 0.0, "neutral"
        
    avg_v = total_v / valid_chunks
    avg_a = total_a / valid_chunks
    avg_d = total_d / valid_chunks
    
    # Determine which of the 28 emotions had the highest cumulative score
    dominant_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else "neutral"
    
    return avg_v, avg_a, avg_d, dominant_emotion

artists_data = {
    "Eminem": ["Recovery", "Music to be Murdered By", "The Death of Slim Shady"],
    "Taylor Swift": ["Speak Now", "The Tortured Poets Department", "The Life of a Showgirl"],
    "Sabrina Carpenter": ["Eyes Wide Open", "Short n' Sweet", "Man's Best Friend"],
    "Kanye West": ["My Beautiful Dark Twisted Fantasy", "Vultures 1", "Vultures 2"],
    "Drake": ["Views", "For All The Dogs", "Some Sexy Songs 4 U"],
    "Kendrick Lamar": ["good kid, m.A.A.d city", "Mr. Morale & the Big Steppers", "GNX"],
    "Juice WRLD": ["Goodbye & Good Riddance", "Fighting Demons", "The Party Never Ends"],
    "Billie Eilish": ["WHEN WE ALL FALL ASLEEP, WHERE DO WE GO?", "Happier Than Ever", "HIT ME HARD AND SOFT"],
    "The Weeknd": ["Kiss Land", "Dawn FM", "Hurry Up Tomorrow"],
    "Green Day": ["¡Uno!", "Father of All Motherfuckers", "Saviors"],
    "J. Cole": ["Born Sinner", "The Off-Season", "Might Delete Later"],
    "Mac Miller": ["Blue Slide Park", "Circles", "Balloonerism"],
    "Playboi Carti": ["Die Lit", "Whole Lotta Red", "MUSIC"],
    "Tame Impala": ["Lonerism", "The Slow Rush", "Deadbeat"]
}

def main():
    # create Song CSV with new name
    if not os.path.exists(SONG_FILE):
        with open(SONG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Artist', 'Album', 'Title', 'Valence', 'Arousal', 'Dominance', 'Primary_Emotion'])

    # create Album CSV with new name
    if not os.path.exists(ALBUM_FILE):
        with open(ALBUM_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Artist', 'Album', 'Avg_Valence', 'Avg_Arousal', 'Avg_Dominance', 'Song_Count'])

    for artist_name, albums in artists_data.items():
        print(f"Working on artist: {artist_name}")
        
        for album_name in albums:
            print(f"   Searching for album: {album_name}")
            try:
                album = genius.search_album(album_name, artist_name)
                
                if album:
                    tracks_to_process = []
                    if hasattr(album, 'songs'): tracks_to_process = album.songs
                    elif hasattr(album, 'tracks'): tracks_to_process = album.tracks
                    
                    print(f"      Found {len(tracks_to_process)} songs. Running GoEmotions analysis...")

                    album_valence, album_arousal, album_dominance = [], [], []

                    for item in tracks_to_process:
                        if isinstance(item, tuple): _, track = item
                        else: track = item

                        if hasattr(track, 'lyrics'): song_lyrics = track.lyrics; title = track.title
                        elif isinstance(track, dict): song_lyrics = track.get('lyrics', ""); title = track.get('song', {}).get('title', "Unknown")
                        else: continue
                        
                        if not song_lyrics: continue
                        clean_lyrics = song_lyrics.split('Lyrics', 1)[-1] if 'Lyrics' in song_lyrics else song_lyrics
                        
                        # Run the complex analysis
                        v, a, d, dom_emo = analyze_lyrics(clean_lyrics)
                        
                        album_valence.append(v)
                        album_arousal.append(a)
                        album_dominance.append(d)

                        print(f"      > {title} | V:{v:.2f} A:{a:.2f} D:{d:.2f} | {dom_emo}")

                        with open(SONG_FILE, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([artist_name, album_name, title, v, a, d, dom_emo])

                    if len(album_valence) > 0:
                        avg_v = statistics.mean(album_valence)
                        avg_a = statistics.mean(album_arousal)
                        avg_d = statistics.mean(album_dominance)
                        
                        with open(ALBUM_FILE, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([artist_name, album_name, avg_v, avg_a, avg_d, len(album_valence)])
                        print(f"      Album Summary Saved.")
                else:
                    print(f"      Could not find album: {album_name}")
            except Exception as e:
                print(f"      Error: {e}")
                time.sleep(2)

if __name__ == "__main__":
    main()  