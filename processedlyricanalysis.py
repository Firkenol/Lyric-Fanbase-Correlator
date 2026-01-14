import pandas as pd
import os
import csv
from transformers import pipeline
from tqdm import tqdm

# Files
input_file = "lyrics_dataset_nlp_processed.csv"
song_output = "song_level_empathetic.csv"
album_output = "album_level_empathetic.csv"

# Model: Trained on the EmpatheticDialogues dataset (Rashkin et al., 2019)
# 32 Labels including: Nostalgic, Sentimental, Furious, Devastated, Yearning
model_name = "bdotloh/distilbert-base-uncased-empathetic-dialogues-context"

def main():
    if not os.path.exists(input_file):
        print(f"File {input_file} not found.")
        return

    # Load data
    df = pd.read_csv(input_file)
    df = df.dropna(subset=['Processed_Lyrics'])
    df = df[df['Processed_Lyrics'].str.len() > 10]

    print(f"Loading EmpatheticDialogues model ({model_name})...")
    # top_k=1 gives the single best fit from the 32 labels
    classifier = pipeline("text-classification", model=model_name, top_k=1)
    
    song_rows = []
    album_emotions = {} 

    print("Analyzing lyrics...")
    
    # Process
    try:
        iterator = tqdm(df.iterrows(), total=len(df))
    except ImportError:
        iterator = df.iterrows()
    
    for index, row in iterator:
        text = str(row['Processed_Lyrics'])
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        
        chunk_results = []
        
        for chunk in chunks:
            try:
                # Get the label defined by the model (no manual filtering needed)
                res = classifier(chunk)[0][0]
                chunk_results.append(res['label'])
            except:
                continue
        
        if not chunk_results: continue

        # Majority Vote
        primary_emo = max(set(chunk_results), key=chunk_results.count)
        
        song_rows.append([
            row['Artist'], 
            row['Album'], 
            row['Title'], 
            primary_emo
        ])
        
        key = (row['Artist'], row['Album'])
        if key not in album_emotions:
            album_emotions[key] = []
        album_emotions[key].append(primary_emo)

    # Save Results
    with open(song_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Artist', 'Album', 'Title', 'Emotion'])
        writer.writerows(song_rows)
    print(f"Song data saved to {song_output}")

    # Album Summary
    with open(album_output, 'w', newline='', encoding='utf-8') as f:
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

    print(f"Album data saved to {album_output}")

if __name__ == "__main__":
    main()