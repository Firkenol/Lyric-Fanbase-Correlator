import pandas as pd
import statistics
import os
import csv
from transformers import pipeline
from tqdm import tqdm

# --- CONFIGURATION ---
# Input: The file we created in the cleaning step
INPUT_FILE = "lyrics_dataset_nlp_processed.csv"

# Outputs: The final scores
SONG_FILE = "song_level_jhartmann_vad.csv"
ALBUM_FILE = "album_level_jhartmann_vad.csv"

# 1. LOAD MODEL
print("Loading j-hartmann model...")
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# 2. VAD MAP
vad_map = {
    'admiration': [0.8, 0.4, -0.2], 'amusement': [0.7, 0.5, 0.5], 'excitement': [0.9, 0.9, 0.7],
    'joy': [0.95, 0.7, 0.8], 'love': [0.9, 0.6, 0.6], 'pride': [0.8, 0.7, 0.9], 
    'optimism': [0.7, 0.5, 0.7], 'gratitude': [0.8, 0.3, -0.1],
    'anger': [-0.7, 0.9, 0.8], 'annoyance': [-0.5, 0.6, 0.4], 'disapproval': [-0.6, 0.5, 0.6],
    'fear': [-0.7, 0.9, -0.6], 'nervousness': [-0.5, 0.8, -0.7],
    'remorse': [-0.8, 0.2, -0.5], 'sadness': [-0.9, -0.2, -0.6], 'disappointment': [-0.7, 0.1, -0.4],
    'embarrassment': [-0.6, 0.5, -0.7], 'grief': [-0.9, 0.2, -0.5],
    'confusion': [-0.3, 0.6, -0.4], 'curiosity': [0.4, 0.6, 0.1], 'realization': [0.2, 0.5, 0.3],
    'surprise': [0.5, 0.9, 0.1], 'neutral': [0.0, 0.0, 0.0],
    'caring': [0.7, 0.2, 0.4], 'desire': [0.6, 0.8, 0.4], 'relief': [0.6, -0.3, 0.2],
    'approval': [0.6, 0.3, 0.5], 'disgust': [-0.8, 0.6, 0.5]
}

def analyze_lyrics(text):
    # Safety check for empty lyrics
    if not isinstance(text, str) or len(text.strip()) < 5:
        return 0.0, 0.0, 0.0, "neutral"

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
            
            for res in results:
                lbl = res['label']
                prob = res['score']
                
                emotion_scores[lbl] = emotion_scores.get(lbl, 0) + prob
                
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
    
    # Soft Neutral Filter
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    winner, winner_score = sorted_emotions[0]
    
    if winner == 'neutral' and len(sorted_emotions) > 1:
        runner_up, runner_score = sorted_emotions[1]
        if runner_score > (winner_score * 0.6):
            dominant_emotion = runner_up
        else:
            dominant_emotion = winner
    else:
        dominant_emotion = winner
    
    return avg_v, avg_a, avg_d, dominant_emotion

def main():
    # 3. READ LOCAL FILE (Instead of Genius)
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run the cleaning script first.")
        return

    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Dictionary to hold album data for aggregation
    # Format: {(Artist, Album): {'v':[], 'a':[], 'd':[], 'emo':[]}}
    album_data = {}

    # 4. PROCESS SONGS
    print("Starting VAD Analysis...")
    
    with open(SONG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Artist', 'Album', 'Title', 'Valence', 'Arousal', 'Dominance', 'Primary_Emotion'])
        
        # Iterate through the DataFrame
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            
            # Use 'Processed_Lyrics' from the cleaning step
            lyrics = row['Processed_Lyrics']
            
            v, a, d, emo = analyze_lyrics(lyrics)
            
            # Write Song Result
            writer.writerow([row['Artist'], row['Album'], row['Title'], v, a, d, emo])
            
            # Add to Album Aggregator
            key = (row['Artist'], row['Album'])
            if key not in album_data:
                album_data[key] = {'v': [], 'a': [], 'd': [], 'emotions': []}
            
            album_data[key]['v'].append(v)
            album_data[key]['a'].append(a)
            album_data[key]['d'].append(d)
            album_data[key]['emotions'].append(emo)

    print(f"Song-level data saved to {SONG_FILE}")

    # 5. PROCESS ALBUMS
    print("Calculating Album Averages...")
    
    with open(ALBUM_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Artist', 'Album', 'Avg_Valence', 'Avg_Arousal', 'Avg_Dominance', 'Album_Primary_Emotion', 'Song_Count'])
        
        for (artist, album), stats in album_data.items():
            count = len(stats['v'])
            if count == 0: continue
            
            avg_v = statistics.mean(stats['v'])
            avg_a = statistics.mean(stats['a'])
            avg_d = statistics.mean(stats['d'])
            
            # Calculate Mode (Most common emotion)
            try:
                primary_emo = statistics.mode(stats['emotions'])
            except:
                # If tie, pick the most frequent one manually
                primary_emo = max(set(stats['emotions']), key=stats['emotions'].count)

            writer.writerow([artist, album, avg_v, avg_a, avg_d, primary_emo, count])
            print(f"   > {album}: {primary_emo.upper()} (V:{avg_v:.2f} A:{avg_a:.2f} D:{avg_d:.2f})")

    print(f"Album-level summaries saved to {ALBUM_FILE}")

if __name__ == "__main__":
    main()