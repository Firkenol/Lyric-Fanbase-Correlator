import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

check_dirs = [
    r"D:\Lyrics-Fanbase-Correlator\Final_Analysis_Results",
    r"D:\Lyrics-Fanbase-Correlator\Processed_Artist_Data"
]

print("HUNTING DOWN THE FILES...\n")

for d in check_dirs:
    if not os.path.exists(d):
        continue
    print(f"--- Checking Folder: {os.path.basename(d)} ---")
    
    for f in os.listdir(d):
        # target the problem artists
        if any(x in f for x in ['Kanye', 'MacMiller', 'Carti', 'Drake', 'Weeknd']):
            path = os.path.join(d, f)
            try:
                # read it just like the master script does
                df = pd.read_csv(path, low_memory=False, on_bad_lines='skip')
                cols = list(df.columns)
                
                has_ai = 'joy' in cols and 'anger' in cols
                
                date_col = next((c for c in ['Date', 'created_utc', 'timestamp'] if c in cols), None)
                if date_col:
                    # test the exact parser the master script uses
                    df['Parsed_Date'] = pd.to_datetime(
                        df[date_col], 
                        errors='coerce', 
                        unit='s' if df[date_col].dtype != 'object' else None
                    )
                    min_date = df['Parsed_Date'].min()
                    max_date = df['Parsed_Date'].max()
                    date_str = f"{min_date} to {max_date}"
                else:
                    date_str = "NO DATE COLUMN FOUND"

                print(f"File: {f}")
                print(f"  -> Rows: {len(df)}")
                print(f"  -> Has AI Columns: {has_ai}")
                print(f"  -> Date Range: {date_str}\n")
                
            except Exception as e:
                print(f"File: {f} | ERROR READING FILE: {e}\n")