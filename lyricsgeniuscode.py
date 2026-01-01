import lyricsgenius
import time
from dotenv import load_dotenv
import os

load_dotenv()
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
genius.verbose = False  # Turn off status messages
genius.remove_section_headers = True 
genius.skip_non_songs = True
genius.excluded_terms = ["(Remix)", "(Live)"] #Filter out non-studio tracks

def get_lyrics_for_artist(artist_name, max_songs=5):
    try:
        print(f"🎤 Searching for {artist_name}...")
        artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
        
        if artist:
            print(f"\nFound {len(artist.songs)} songs for {artist.name}:")
            for song in artist.songs:
                print(f"   - {song.title}")
                #print(song.lyrics[:100]) 
            
            
            artist.save_lyrics()
            print(f"\nLyrics saved to {artist_name.replace(' ', '')}.json")
            return artist
        else:
            print("Artist not found.")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    carti_data = get_lyrics_for_artist("Playboi Carti", max_songs=5)