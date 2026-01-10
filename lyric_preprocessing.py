print("Script is starting...") # DEBUG LINE

import pandas as pd
import re
import os
import sys

# We import spacy but DON'T load the model yet
import spacy
from collections import Counter

# CONFIGURATION
INPUT_FILE = "lyrics_dataset.csv"
OUTPUT_FILE = "lyrics_dataset_nlp_processed.csv"

# KEY CHANGE: Customize stop words to exclude negations
# We want these words to stay in the text because they flip sentiment
negation_words = {"no", "not", "never", "none", "nothing", "neither", "nor", "nowhere", "cannot", "cant", "wont"}

# Remove them from Spacy's default stop word list so is_stop returns False for them
for word in negation_words:
    nlp.vocab[word].is_stop = False

# 1. Semantic Noise Filter
VOCABLES = [
    r"\b(woah|whoa|oh|ooh|ah|ahh|uh|uhh|hmm|hm|mmm)\b", 
    r"\b(la|da|na|di|doo|dum|dududu|ba|bum|du)\b", 
    r"\b(ay|ayy|yuh|yah|yeah|yeh|yea)\b", 
    r"\b(skrrt|skrt|grrt|brrt|bow|pow|phew)\b", 
    r"\b(ha|haha|hahaha|heh)\b", 
    r"\b(yo|hey|huh|what|nah|nanana)\b"
]

# 2. Specific Song Purges
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

# 3. Global Metadata Bans
GLOBAL_BANS = [
    "ticket master", "produced by", "lyrics", "verse", "chorus", "hook", 
    "intro", "outro", "bridge", "instrumental", "skit", "interlude",
    "embed", "share", "url", "copy", "click to", "sign up"
]

def load_spacy_safe():
    print("Loading Spacy Model (this might take a moment)...")
    try:
        # Disable parser/ner for speed
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        print("Spacy Loaded Successfully.")
        return nlp
    except OSError:
        print("Error: Spacy model 'en_core_web_sm' not found.")
        print("Run: python -m spacy download en_core_web_sm")
        sys.exit()

def nlp_clean_logic(row, nlp_model):
    text = str(row['Lyrics'])
    album = row['Album']
    title = row['Title']
    
    # Phase 1: Structure & Manual Cleanup
    if (album, title) in SPECIFIC_PURGES:
        targets = SPECIFIC_PURGES[(album, title)]
        if targets == ["DELETE_SONG"]:
            return ""
        for target in targets:
            text = re.sub(re.escape(target), "", text, flags=re.IGNORECASE)

    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\{.*?\}', ' ', text)
    text = re.sub(r'\d+\s?Contributors.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Embed$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+)-\1\b', r'\1', text, flags=re.IGNORECASE) 

    for pattern in VOCABLES:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        
    # Phase 2: Spacy NLP
    doc = nlp_model(text)
    clean_tokens = []
    
    for token in doc:
        if token.is_stop: continue
        if token.is_punct or token.like_num: continue
        
        lemma = token.lemma_.lower().strip()
        if len(lemma) < 2: continue
            
        clean_tokens.append(lemma)
        
    text = " ".join(clean_tokens)

    # Phase 3: Repetition Reduction
    lines = text.split('\n')
    unique_lines = []
    prev_line = ""
    for line in lines:
        clean_line = line.strip()
        if not clean_line: continue
        if clean_line == prev_line: continue
        unique_lines.append(clean_line)
        prev_line = clean_line

    return " ".join(unique_lines)

def main():
    # DEBUG: Check file path
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, INPUT_FILE)
    print(f"Looking for input file at: {file_path}")

    if not os.path.exists(file_path):
        print("ERROR: lyrics_dataset.csv NOT FOUND.")
        print("Make sure the file is in the same folder where you are running the command.")
        return

    print("Reading CSV...")
    df = pd.read_csv(file_path)

    # Load Spacy NOW (inside main)
    nlp = load_spacy_safe()

    print("Running NLP Pipeline on lyrics...")
    
    # Pass nlp object to the function
    try:
        from tqdm import tqdm
        tqdm.pandas()
        df['Processed_Lyrics'] = df.progress_apply(lambda row: nlp_clean_logic(row, nlp), axis=1)
    except ImportError:
        print("(Install 'tqdm' for a progress bar next time)")
        df['Processed_Lyrics'] = df.apply(lambda row: nlp_clean_logic(row, nlp), axis=1)

    # Filter
    initial_count = len(df)
    df = df[df['Processed_Lyrics'].str.len() > 5] 
    final_count = len(df)
    
    print(f"\Done. Removed {initial_count - final_count} empty tracks.")

    # Save
    df[['Artist', 'Album', 'Title', 'Processed_Lyrics']].to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()