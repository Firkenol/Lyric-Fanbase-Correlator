import os
import csv
import sys
import lyricsgenius
import pandas as pd
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. HEALTH CHECK ---
print(f"🔍 Using 'lyricsgenius' from: {lyricsgenius.__file__}")
# If this prints a path in your 'd:\Lyrics-Fanbase-Correlator' folder, 
# DELETE that file (e.g., lyricsgenius.py) immediately. 
# It should point to '...site-packages/lyricsgenius/...'

# --- 2. SETUP ---
load_dotenv()
token = os.getenv("GENIUS_TOKEN")

# Initialize
genius = lyricsgenius.Genius(token)
genius.verbose = False 
genius.remove_section_headers = True
genius.skip_non_songs = True
analyzer = SentimentIntensityAnalyzer()

OUTPUT_FILE = "lyrics_analysis_results.csv"

# Your Data
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
    "Green Day": ["¡Uno!", "Father of All...", "Saviors"],
    "J. Cole": ["Born Sinner", "The Off-Season", "Might Delete Later"],
    "Mac Miller": ["Blue Slide Park", "Circles", "Balloonerism"],
    "Playboi Carti": ["Die Lit", "Whole Lotta Red", "MUSIC"],
    "Tame Impala": ["Lonerism", "The Slow Rush", "Deadbeat"]
}

def analyze_lyrics(lyrics):
    return analyzer.polarity_scores(lyrics)

# --- 3. THE SAFE SCRAPER ---
def main():
    # Create CSV if missing
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Artist', 'Album', 'Song_Title', 'Word_Count', 'Sentiment_Compound', 'Sentiment_Pos', 'Sentiment_Neg', 'Sentiment_Neu'])

    for artist_name, albums in artists_data.items():
        print(f"\n🎤 Working on artist: {artist_name}")
        
        for album_name in albums:
            print(f"   💿 Searching for album: {album_name}")
            
            try:
                # Search for the album
                album = genius.search_album(album_name, artist_name)
                
                if album:
                    # --- CRITICAL FIX: CHECK ATTRIBUTES ---
                    # We check where the songs are hidden in this object
                    tracks_to_process = []
                    
                    if hasattr(album, 'songs'):
                        tracks_to_process = album.songs
                    elif hasattr(album, 'tracks'):
                        # Sometimes it's called 'tracks' and works differently
                        print("      ⚠️ Note: Found '.tracks' instead of '.songs'")
                        tracks_to_process = album.tracks
                    else:
                        print(f"      ❌ Error: Album object has unknown structure. Keys: {dir(album)}")
                        continue

                    print(f"      ✅ Found {len(tracks_to_process)} songs.")

                    for track in tracks_to_process:
                        # Handle different track object types
                        if hasattr(track, 'lyrics'):
                            song_lyrics = track.lyrics
                            title = track.title
                        elif isinstance(track, dict):
                            # Sometimes it's a raw dictionary
                            song_lyrics = track.get('lyrics', "")
                            title = track.get('song', {}).get('title', "Unknown")
                        else:
                            continue

                        if not song_lyrics: continue # Skip if empty

                        # Clean Title from Lyrics
                        clean_lyrics = song_lyrics.split('Lyrics', 1)[-1] if 'Lyrics' in song_lyrics else song_lyrics
                        
                        # NLP
                        sentiment = analyze_lyrics(clean_lyrics)
                        words = clean_lyrics.split()
                        
                        # Save
                        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                artist_name, 
                                album_name, 
                                title, 
                                len(words), 
                                sentiment['compound'], 
                                sentiment['pos'], 
                                sentiment['neg'], 
                                sentiment['neu']
                            ])
                else:
                    print(f"      ❌ Could not find album: {album_name}")
            
            except Exception as e:
                print(f"      ⚠️ Error processing {album_name}: {e}")

if __name__ == "__main__":
    main()