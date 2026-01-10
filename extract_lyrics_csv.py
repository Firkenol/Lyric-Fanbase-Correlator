import os
import csv
import sys
import time
from dotenv import load_dotenv
import lyricsgenius

# Load token
load_dotenv()
TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
if not TOKEN:
    print("Error: GENIUS_ACCESS_TOKEN not found.")
    sys.exit(1)

# Initialize Genius
genius = lyricsgenius.Genius(TOKEN, timeout=25, retries=3)
genius.verbose = False
genius.remove_section_headers = True
genius.skip_non_songs = True

OUTPUT_FILE = "song_lyrics_only.csv"

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


def ensure_output():
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["Artist", "Album", "Title", "Lyrics"])


def extract_and_save():
    ensure_output()

    for artist_name, albums in artists_data.items():
        print(f"Working on artist: {artist_name}")
        for album_name in albums:
            print(f"  Searching album: {album_name}")
            try:
                album = genius.search_album(album_name, artist_name)
                if not album:
                    print(f"    Album not found: {album_name}")
                    continue

                tracks = []
                if hasattr(album, 'songs'):
                    tracks = album.songs
                elif hasattr(album, 'tracks'):
                    tracks = album.tracks

                print(f"    Found {len(tracks)} tracks (approx).")

                for item in tracks:
                    # some results come as (index, track) tuples
                    track = item[1] if isinstance(item, tuple) else item

                    # extract lyrics and title depending on object type
                    lyrics = ""
                    title = "Unknown"
                    if hasattr(track, 'lyrics'):
                        lyrics = track.lyrics or ""
                        title = getattr(track, 'title', title)
                    elif isinstance(track, dict):
                        lyrics = track.get('lyrics', '')
                        # nested song title for some dicts
                        title = track.get('song', {}).get('title', track.get('title', title))

                    if not lyrics:
                        continue

                    # remove any leading 'Lyrics' header that Genius may include
                    clean_lyrics = lyrics.split('Lyrics', 1)[-1] if 'Lyrics' in lyrics else lyrics

                    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                        writer.writerow([artist_name, album_name, title, clean_lyrics])

                    print(f"      Saved: {title}")

            except Exception as e:
                print(f"    Error fetching album '{album_name}' for '{artist_name}': {e}")
                time.sleep(1)


if __name__ == "__main__":
    extract_and_save()
    print(f"Done. Lyrics saved to {OUTPUT_FILE}")
