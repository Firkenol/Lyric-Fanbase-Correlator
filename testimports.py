import sys

try:
    import lyricsgenius
    print("lyricsgenius is installed and working!")
except ImportError:
    print("lyricsgenius FAILED to import.")

try:
    from dotenv import load_dotenv
    print("python-dotenv is installed and working!")
except ImportError:
    print("python-dotenv FAILED to import.")
    
print("\nPython is looking for packages here:")
for path in sys.path:
    print(path)