import pandas as pd
import glob

# Path to your CSV files
file_pattern = "../outputs/responses/py_dev_mini_1/responses_*.csv"  # adjust folder path if needed

# Get a list of all matching CSV files
csv_files = glob.glob(file_pattern)

# Read all CSV files and combine them
dfs = [pd.read_csv(f) for f in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)

# Optional: sort by timestamp if you want chronological order
combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
combined_df = combined_df.sort_values('timestamp')

# Save to a new CSV file
combined_df.to_csv("combined_responses.csv", index=False)
