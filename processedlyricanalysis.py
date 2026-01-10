import pandas as pd
import os
import csv
from transformers import pipeline
from tqdm import tqdm

# Files
INPUT_FILE = "lyrics_dataset_nlp_processed.csv"
SONG_OUTPUT = "song_level_goemotions_filtered.csv"
ALBUM_OUTPUT = "album_level_goemotions_filtered.csv"

# Model
MODEL_NAME = "SamLowe/roberta-base-go_emotions"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found.")
        return

    # Load data
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=['Processed_Lyrics'])

    # Load model
    # top_k=None ensures we get all scores so we can manually filter Neutral
    classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None)
    
    song_rows = []
    album_emotions = {} 

    print("Analyzing with Polarity Filtering...")
    
    # Iterator
    try:
        iterator = tqdm(df.iterrows(), total=len(df))
    except ImportError:
        iterator = df.iterrows()
    
    for index, row in iterator:
        text = str(row['Processed_Lyrics'])
        # Chunking
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        
        song_scores = {}
        
        for chunk in chunks:
            try:
                results = classifier(chunk)[0]
                for res in results:
                    label = res['label']
                    score = res['score']
                    
                    # FILTERING LOGIC: Ignore Neutral completely
                    # This forces the model to pick the strongest 'real' emotion
                    if label == 'neutral': 
                        continue
                        
                    song_scores[label] = song_scores.get(label, 0) + score
            except:
                continue
        
        # If the song was 100% neutral (rare), fall back to neutral
        if not song_scores:
            final_emotion = "neutral"
        else:
            # Pick highest cumulative score among non-neutral emotions
            final_emotion = max(song_scores, key=song_scores.get)

        song_rows.append([row['Artist'], row['Album'], row['Title'], final_emotion])
        
        # Album Stats
        key = (row['Artist'], row['Album'])
        if key not in album_emotions: album_emotions[key] = []
        album_emotions[key].append(final_emotion)

    # Save Song
    with open(SONG_OUTPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Artist', 'Album', 'Title', 'Primary_Emotion'])
        writer.writerows(song_rows)
    print(f"Song results saved to {SONG_OUTPUT}")

    # Save Album
    with open(ALBUM_OUTPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Artist', 'Album', 'Dominant_Emotion', 'Breakdown', 'Song_Count'])
        
        for (artist, album), emotion_list in album_emotions.items():
            if not emotion_list: continue
            
            counts = {}
            for emo in emotion_list:
                counts[emo] = counts.get(emo, 0) + 1
            
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            dominant = sorted_counts[0][0]
            summary = ", ".join([f"{e} ({c})" for e, c in sorted_counts[:3]])
            
            writer.writerow([artist, album, dominant, summary, len(emotion_list)])
            print(f"> {album}: {dominant.upper()} [{summary}]")

    print(f"Album results saved to {ALBUM_OUTPUT}")

if __name__ == "__main__":
    main()