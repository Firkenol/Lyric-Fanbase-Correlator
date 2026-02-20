import pandas as pd
import re
import os
import sys
import spacy
import csv
import numpy as np
import statistics
from transformers import pipeline
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "lyrics_dataset.csv"
SONG_OUTPUT = "song_level_final_complex.csv"
ALBUM_OUTPUT = "album_level_final_complex.csv"

# 1. THE 28-EMOTION DICTIONARY
# This acts as the "Target Map". We calculate which of these points 
# the song's VAD score is closest to.
VAD_MAP = {
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

# 2. NOISE FILTER LISTS
VOCABLES = [
    r"\b(woah|whoa|oh|ooh|ah|ahh|uh|uhh|hmm|hm|mmm)\b", 
    r"\b(la|da|na|di|doo|dum|dududu|ba|bum|du)\b", 
    r"\b(ay|ayy|yuh|yah|yeah|yeh|yea)\b", 
    r"\b(skrrt|skrt|grrt|brrt|bow|pow|phew)\b", 
    r"\b(ha|haha|hahaha|heh)\b", 
    r"\b(yo|hey|huh|what|nah|nanana)\b"
]

SPECIFIC_PURGES = {
    ("My Beautiful Dark Twisted Fantasy", "Runaway"): ["look at ya", "ladies and gentlemen"],
    ("My Beautiful Dark Twisted Fantasy", "Power"): ["21st century schizoid man"],
    ("My Beautiful Dark Twisted Fantasy", "Monster"): ["gossip, gossip", "f-u"],
    ("My Beautiful Dark Twisted Fantasy", "Blame Game"): ["chris rock", "yeezy taught me"], 
    ("My Beautiful Dark Twisted Fantasy", "Who Will Survive in America"): ["DELETE_SONG"], 
    ("My Beautiful Dark Twisted Fantasy", "See Me Now"): ["DELETE_SONG"],
    ("Vultures 1", "Hoodrat"): ["hoodrat", "whore"],
    ("Vultures 1", "Beg Forgiveness"): ["oh-ah-ah"],
    ("Vultures 1", "Keys To My Life"): ["m.o"],
    ("Vultures 1", "Paid"): ["fri-fri", "ai-ai-ai-aid"],
    ("Vultures 2", "530"): ["da-da", "na-dana", "pa-da-la", "fa-na-dan", "sunna-wunna"],
    ("Vultures 2", "Isabella"): ["DELETE_SONG"],
    ("Vultures 2", "Field Trip"): ["nah-nah-nah"],
    ("Die Lit", "Pull Up"): ["pull up"],
    ("Die Lit", "Lean 4 Real"): ["sus", "what"],
    ("Whole Lotta Red", "JumpOutTheHouse"): ["jump out the house"],
    ("Whole Lotta Red", "Teen X"): ["cough syrup"],
    ("Recovery", "Cold Wind Blows"): ["dum, du-du-du-dum"],
    ("To Pimp a Butterfly", "For Free?"): ["this dick ain't free"]
}

# Negation words to explicitly PROTECT from being deleted
NEGATION_WORDS = {"no", "not", "never", "none", "nothing", "neither", "nor", "nowhere", "cannot", "cant", "wont"}

# --- PART 1: PRE-PROCESSING FUNCTIONS ---

def init_spacy():
    print("Loading Spacy and configuring negation protection...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        # Customizing stop words to ensure negations are NOT removed
        for word in NEGATION_WORDS:
            if word in nlp.vocab:
                nlp.vocab[word].is_stop = False
        return nlp
    except OSError:
        print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
        sys.exit()

def clean_text(row, nlp):
    text = str(row['Lyrics'])
    album, title = row['Album'], row['Title']
    
    # Check specific purges
    if (album, title) in SPECIFIC_PURGES:
        targets = SPECIFIC_PURGES[(album, title)]
        if targets == ["DELETE_SONG"]: return ""
        for target in targets: text = re.sub(re.escape(target), "", text, flags=re.IGNORECASE)

    # General Regex Cleaning
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\{.*?\}', ' ', text)
    text = re.sub(r'\d+\s?Contributors.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Embed$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+)-\1\b', r'\1', text, flags=re.IGNORECASE) 

    for p in VOCABLES: text = re.sub(p, " ", text, flags=re.IGNORECASE)
        
    # NLP Tokenization
    doc = nlp(text)
    
    # Keep tokens that are NOT stops, punct, or numbers.
    # UNLESS they are in NEGATION_WORDS (which are now marked not-stop).
    tokens = [t.lemma_.lower().strip() for t in doc if not t.is_stop and not t.is_punct and not t.like_num]
    
    # Size filter: Delete single letters unless they are negation words (like 'no')
    tokens = [t for t in tokens if len(t) > 1 or t in NEGATION_WORDS]
    
    # Repetition Reduction (remove identical consecutive lines)
    unique_lines = []
    prev_line = ""
    for line in " ".join(tokens).split('\n'):
        clean = line.strip()
        if clean == prev_line: continue
        unique_lines.append(clean)
        prev_line = clean
        
    return " ".join(unique_lines)

# --- PART 2: VAD TRANSLATION FUNCTIONS ---

def get_complex_emotion(v, a, d):
    # This function finds the nearest neighbor in the 3D VAD space
    song_coords = np.array([v, a, d])
    min_dist = float('inf')
    best_emotion = "neutral"
    
    for emotion, coords in VAD_MAP.items():
        dist = np.linalg.norm(song_coords - np.array(coords))
        if dist < min_dist:
            min_dist = dist
            best_emotion = emotion
            
    return best_emotion

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. RUN CLEANING
    nlp = init_spacy()
    print(f"Reading {INPUT_FILE} and cleaning text...")
    
    df = pd.read_csv(INPUT_FILE)
    
    # Apply cleaning
    # using simple apply instead of tqdm for compatibility if tqdm is missing
    try:
        tqdm.pandas(desc="Cleaning")
        df['Clean_Lyrics'] = df.progress_apply(lambda row: clean_text(row, nlp), axis=1)
    except:
        df['Clean_Lyrics'] = df.apply(lambda row: clean_text(row, nlp), axis=1)

    # Remove empty rows
    initial_len = len(df)
    df = df[df['Clean_Lyrics'].str.len() > 5]
    print(f"Cleaning complete. Removed {initial_len - len(df)} unreadable tracks.")

    # 2. LOAD MODEL
    print("Loading J-Hartmann Model...")
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    
    album_data = {} # Aggregator

    print("Starting Analysis (VAD Calculation + Complex Translation)...")
    
    with open(SONG_OUTPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Artist', 'Album', 'Title', 'Valence', 'Arousal', 'Dominance', 'Complex_Emotion'])
        
        # Iterate and Process
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
            text = row['Clean_Lyrics']
            
            # Chunking because BERT models have a 512 token limit
            chunks = [text[i:i+512] for i in range(0, len(text), 512)]
            
            total_v, total_a, total_d = 0, 0, 0
            valid_chunks = 0
            
            for chunk in chunks:
                if len(chunk) < 10: continue
                try:
                    # Get model probabilities for the 7 basic emotions
                    results = classifier(chunk)[0]
                    
                    # Convert those 7 basic probs into VAD coordinates
                    # using the coordinates of the 7 basic emotions in our map
                    chunk_v, chunk_a, chunk_d = 0, 0, 0
                    
                    for res in results:
                        label = res['label'] # e.g., 'joy', 'sadness'
                        score = res['score']
                        
                        # We only use the dictionary keys that match the model's output
                        if label in VAD_MAP:
                            coords = VAD_MAP[label]
                            chunk_v += coords[0] * score
                            chunk_a += coords[1] * score
                            chunk_d += coords[2] * score
                            
                    total_v += chunk_v
                    total_a += chunk_a
                    total_d += chunk_d
                    valid_chunks += 1
                except: continue
            
            if valid_chunks > 0:
                avg_v = total_v / valid_chunks
                avg_a = total_a / valid_chunks
                avg_d = total_d / valid_chunks
                
                # TRANSLATE: Convert averaged VAD -> 28 Complex Emotions
                complex_emo = get_complex_emotion(avg_v, avg_a, avg_d)
                
                writer.writerow([row['Artist'], row['Album'], row['Title'], avg_v, avg_a, avg_d, complex_emo])
                
                # Save for album stats
                key = (row['Artist'], row['Album'])
                if key not in album_data: 
                    album_data[key] = {'v':[], 'a':[], 'd':[], 'emotions':[]}
                
                album_data[key]['v'].append(avg_v)
                album_data[key]['a'].append(avg_a)
                album_data[key]['d'].append(avg_d)
                album_data[key]['emotions'].append(complex_emo)

    # 3. ALBUM AGGREGATION
    print("Generating Album Summaries...")
    with open(ALBUM_OUTPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Artist', 'Album', 'Avg_Valence', 'Avg_Arousal', 'Avg_Dominance', 'Primary_Emotion', 'Song_Count'])
        
        for (artist, album), stats in album_data.items():
            if not stats['v']: continue
            
            avg_v = statistics.mean(stats['v'])
            avg_a = statistics.mean(stats['a'])
            avg_d = statistics.mean(stats['d'])
            
            # Determine primary emotion by Mode (most frequent)
            # If tie, recalculate nearest neighbor based on album average VAD
            try:
                primary = statistics.mode(stats['emotions'])
            except:
                primary = get_complex_emotion(avg_v, avg_a, avg_d)
                
            writer.writerow([artist, album, avg_v, avg_a, avg_d, primary, len(stats['v'])])
            print(f" > {album}: {primary.upper()}")

    print(f"Done. Song data saved to {SONG_OUTPUT}")
    print(f"Done. Album data saved to {ALBUM_OUTPUT}")

if __name__ == "__main__":
    main()