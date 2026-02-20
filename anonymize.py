import os
import sys
import csv
import json
import tempfile
import shutil
from pathlib import Path

# CONFIGURATION
RAW_DIR_NAME = "raw"
TARGET_KEYS = ["author", "author_fullname", "username"]  # Keys to NUKE

def get_raw_dir():
    # Finds the 'raw' folder relative to this script
    return Path(__file__).resolve().parent / RAW_DIR_NAME

def clean_jsonl_stream(input_path, temp_path):
    """
    Reads JSONL line-by-line, removes author, writes to temp.
    Returns True if successful, False if it wasn't valid JSONL.
    """
    try:
        with open(input_path, 'r', encoding='utf-8', errors='replace') as fin, \
             open(temp_path, 'w', encoding='utf-8', newline='\n') as fout:
            
            first_line = True
            for line in fin:
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    
                    # Heuristic: If the first line isn't a dict, it's probably not the data we want
                    if first_line:
                        if not isinstance(data, dict):
                            return False
                        first_line = False

                    # NUKE THE KEYS
                    for key in TARGET_KEYS:
                        data.pop(key, None)
                    
                    # Write back to temp
                    fout.write(json.dumps(data) + "\n")
                    
                except json.JSONDecodeError:
                    # If the very first line fails, it's not JSONL. Abort.
                    if first_line:
                        return False
                    # If a middle line fails, skip it (corrupt data)
                    continue
        return True
    except Exception as e:
        print(f"  [JSONL Fail] {e}")
        return False

def clean_csv_stream(input_path, temp_path):
    """
    Reads CSV, removes author column, writes to temp.
    """
    try:
        with open(input_path, 'r', encoding='utf-8', errors='replace', newline='') as fin:
            # Sniff or just read headers
            sample = fin.read(2048)
            fin.seek(0)
            
            # If it doesn't look like CSV, abort
            try:
                if not csv.Sniffer().has_header(sample):
                    return False
            except csv.Error:
                pass # Proceed cautiously if sniffer fails, usually DictReader handles it

            reader = csv.DictReader(fin)
            if not reader.fieldnames:
                return False
            
            # Filter out the target keys from headers
            new_headers = [h for h in reader.fieldnames if h not in TARGET_KEYS]
            
            # If no headers changed, maybe 'author' wasn't there. We still process to be safe.
            
            with open(temp_path, 'w', encoding='utf-8', newline='') as fout:
                writer = csv.DictWriter(fout, fieldnames=new_headers)
                writer.writeheader()
                
                for row in reader:
                    # Create new row without the banned keys
                    clean_row = {k: v for k, v in row.items() if k in new_headers}
                    writer.writerow(clean_row)
        return True
    except Exception as e:
        print(f"  [CSV Fail] {e}")
        return False

def process_file_force(file_path):
    print(f"Processing: {file_path.name}...")
    
    # Create a temp file to write cleaned data to
    fd, temp_path = tempfile.mkstemp(dir=file_path.parent, text=True)
    os.close(fd)
    
    success = False
    
    # STRATEGY 1: Try JSONL (Most likely for Reddit Raw)
    if clean_jsonl_stream(file_path, temp_path):
        print(f"  -> Detected JSONL. Anonymized.")
        success = True
    
    # STRATEGY 2: If JSONL failed, try CSV
    elif clean_csv_stream(file_path, temp_path):
        print(f"  -> Detected CSV. Anonymized.")
        success = True
        
    else:
        print(f"  -> COULD NOT PROCESS (Unknown format). Skipping.")
        os.remove(temp_path)
        return

    # If successful, replace the original file with the temp file
    if success:
        try:
            # Backup original just in case? Uncomment next line if you want safety
            # shutil.move(str(file_path), str(file_path) + ".bak") 
            
            shutil.move(temp_path, file_path)
        except OSError as e:
            print(f"  Error overwriting file: {e}")
            os.remove(temp_path)

def main():
    raw_dir = get_raw_dir()
    
    if not raw_dir.exists():
        print(f"ERROR: Could not find 'raw' directory at: {raw_dir}")
        return

    print(f"Scanning directory: {raw_dir}")
    print("------------------------------------------------")

    files_processed = 0
    
    # Iterate over ALL files in the directory
    for entry in os.scandir(raw_dir):
        if entry.is_file():
            # Skip hidden files or the script itself if it's in there
            if entry.name.startswith("."): 
                continue
            if entry.name.endswith(".py"): 
                continue
            
            process_file_force(Path(entry.path))
            files_processed += 1
            
    print("------------------------------------------------")
    print(f"Done. Processed {files_processed} files.")

if __name__ == "__main__":
    main()