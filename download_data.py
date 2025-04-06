from pybaseball import statcast, playerid_lookup, statcast_batter
import pickle
import os
import time
import pandas as pd

print("Downloading baseball data...")

# Create directory if it doesn't exist
os.makedirs("Suport_Vector_Machine", exist_ok=True)

def download_player_data(player_name, player_id, start_date, end_date, output_file, max_retries=3):
    """Download player data with retry logic and error handling"""
    print(f"\nAttempting to download {player_name} data...")
    retries = 0
    
    while retries < max_retries:
        try:
            print(f"  Download attempt {retries+1} for {player_name} (ID: {player_id})")
            print(f"  Date range: {start_date} to {end_date}")
            
            # Download data
            stats = statcast_batter(start_date, end_date, player_id)
            
            # Check if data is valid
            if stats is None or len(stats) == 0:
                print(f"  Warning: No data returned for {player_name}")
                retries += 1
                time.sleep(5)  # Wait before retry
                continue
                
            print(f"  Successfully downloaded {len(stats)} rows for {player_name}")
            
            # Save to pickle file
            with open(output_file, "wb") as f:
                pickle.dump(stats, f)
            
            print(f"  Saved {player_name} data to {output_file}")
            
            # Create a backup CSV for inspection
            csv_file = output_file.replace('.p', '.csv')
            stats.to_csv(csv_file, index=False)
            print(f"  Also saved CSV backup to {csv_file}")
            
            return True
            
        except Exception as e:
            print(f"  Error downloading {player_name} data: {str(e)}")
            retries += 1
            if retries < max_retries:
                print(f"  Waiting 10 seconds before retry...")
                time.sleep(10)
    
    print(f"  Failed to download {player_name} data after {max_retries} attempts")
    
    # Create empty placeholder file
    empty_df = pd.DataFrame()
    with open(output_file, "wb") as f:
        pickle.dump(empty_df, f)
    print(f"  Created empty placeholder file at {output_file}")
    
    return False

# Download Aaron Judge data
download_player_data(
    "Aaron Judge", 
    592450, 
    '2017-04-02', 
    '2017-10-01', 
    "Suport_Vector_Machine/aaron_judge.p"
)

# Wait between requests to avoid rate limits
time.sleep(10)

# Download Jose Altuve data
download_player_data(
    "Jose Altuve", 
    514888, 
    '2017-04-03', 
    '2017-10-01', 
    "Suport_Vector_Machine/jose_altuve.p"
)

# Wait between requests to avoid rate limits
time.sleep(10)

# Download David Ortiz data
download_player_data(
    "David Ortiz", 
    120074, 
    '2015-04-02', 
    '2015-10-02', 
    "Suport_Vector_Machine/david_ortiz.p"
)

# Check what was successfully downloaded
for player, filename in [
    ("Aaron Judge", "Suport_Vector_Machine/aaron_judge.p"),
    ("Jose Altuve", "Suport_Vector_Machine/jose_altuve.p"),
    ("David Ortiz", "Suport_Vector_Machine/david_ortiz.p")
]:
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
            print(f"\n{player} data: {len(data)} rows")
        except Exception as e:
            print(f"\n{player} data: Error reading file - {str(e)}")
    else:
        print(f"\n{player} data: File not found")

print("\nDownload process completed.")