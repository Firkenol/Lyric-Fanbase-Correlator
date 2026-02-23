import pandas as pd
import os

folder = r"D:\Lyrics-Fanbase-Correlator\Final_Analysis_Results"

for artist in ["Drake", "PlayboiCarti", "MacMiller"]:
    file_path = os.path.join(folder, f"{artist}_FullDist.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, usecols=['Date'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # print the exact timeframe of the scraped data
        print(f"--- {artist} ---")
        print(f"Oldest post: {df['Date'].min()}")
        print(f"Newest post: {df['Date'].max()}\n")