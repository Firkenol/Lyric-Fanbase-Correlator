import pandas as pd
import re
import os
import sys
import spacy
from collections import Counter

# Configuration
INPUT_FILE = "lyrics_dataset.csv"
OUTPUT_FILE = "lyrics_dataset_nlp_processed.csv"

# 1. NOISE FILTERS
VOCABLES = [
    r"\b(woah|whoa|oh|ooh|ah|ahh|uh|uhh|hmm|hm|mmm)\b", 
    r"\b(la|da|na|di|doo|dum|dududu|ba|bum|du)\b", 
    r"\b(ay|ayy|yuh|yah|yeah|yeh|yea)\b", 
    r"\b(skrrt|skrt|grrt|brrt|bow|pow|phew)\b", 
    r"\b(ha|haha|hahaha|heh)\b", 
    r"\b(yo|hey|huh|what|nah|nanana)\b"
]

# 2. SPECIFIC SONG PURGES
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

# 3. NEGATION WORDS TO KEEP (Crucial for VAD)
NEGATION_WORDS = {"no", "not", "never", "none", "nothing", "neither", "nor", "nowhere", "cannot", "cant", "wont"}

def init_spacy_model():
    """Safe loader that handles Windows errors and applies negation logic."""
    print("Loading Spacy Model...")
    try:
        # Load the model
        nlp_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        
        # KEY FIX: Customize the vocabulary *inside* this function
        # We tell Spacy: "Do NOT treat 'not' as a garbage stop word."
        for word in NEGATION_WORDS:
            if word in nlp_model.vocab:
                nlp_model.vocab[word].is_stop = False
        
        print("Model loaded & Negations preserved.")
        return nlp_model
        
    except OSError:
        print("Error: Spacy model not found. Run: python -m spacy download en_core_web_sm")
        sys.exit()

def nlp_clean_logic(row, nlp_model):
    """Cleaning logic that accepts the nlp object as an argument."""
    text = str(row['Lyrics'])
    album = row['Album']
    title = row['Title']
    
    # Phase 1: Manual Purges
    if (album, title) in SPECIFIC_PURGES:
        targets = SPECIFIC_PURGES[(album, title)]
        if targets == ["DELETE_SONG"]:
            return ""
        for target in targets:
            text = re.sub(re.escape(target), "", text, flags=re.IGNORECASE)

    # Regex Cleaning
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\{.*?\}', ' ', text)
    text = re.sub(r'\d+\s?Contributors.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Embed$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+)-\1\b', r'\1', text, flags=re.IGNORECASE) 

    for pattern in VOCABLES:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        
    # Phase 2: Spacy NLP (Using the passed nlp_model)
    doc = nlp_model(text)
    clean_tokens = []
    
    for token in doc:
        # 1. Check Stop Words (Negations are now FALSE here, so they are kept)
        if token.is_stop:
            continue
            
        # 2. Check Punctuation/Numbers
        if token.is_punct or token.like_num:
            continue
            
        # 3. Lemmatize
        lemma = token.lemma_.lower().strip()
        
        # 4. Length check (Keep short words if they are negations like 'no')
        if len(lemma) < 2 and lemma not in NEGATION_WORDS:
            continue
            
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
    # Check file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Initialize Spacy HERE
    nlp = init_spacy_model()

    print("Running Cleaning Pipeline...")
    
    # Pass the 'nlp' object into the function using lambda
    try:
        from tqdm import tqdm
        tqdm.pandas()
        df['Processed_Lyrics'] = df.progress_apply(lambda row: nlp_clean_logic(row, nlp), axis=1)
    except ImportError:
        df['Processed_Lyrics'] = df.apply(lambda row: nlp_clean_logic(row, nlp), axis=1)

    # Filter Empty
    initial_count = len(df)
    df = df[df['Processed_Lyrics'].str.len() > 5] 
    final_count = len(df)
    
    print(f"Removed {initial_count - final_count} empty tracks.")

    # Validate Negations are present
    all_text = " ".join(df['Processed_Lyrics'])
    words = all_text.split()
    print("\nMost Common Words (Check if 'not/no' are here):")
    print(Counter(words).most_common(15))

    # Save
    df[['Artist', 'Album', 'Title', 'Processed_Lyrics']].to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()