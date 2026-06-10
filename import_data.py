import pandas as pd
import zipfile
import io
import urllib.request

print("Downloading dataset from UCI (this might take a minute)...")
url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"

# Download and extract the zip file directly from the web
response = urllib.request.urlopen(url)
with zipfile.ZipFile(io.BytesIO(response.read())) as zipped_file:
    with zipped_file.open('household_power_consumption.txt') as file:
        print("Parsing and cleaning CSV...")
        # Read the raw text file inside the zip
        df = pd.read_csv(file, sep=';', low_memory=False)

print("Preparing 'datetime' column for the backend...")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['datetime'] = df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time']

# We don't need the separated columns anymore
df.drop(columns=['Date', 'Time'], inplace=True)

# Save it where the .env file expects it
output_path = "data/processed_data.csv"
print(f"Saving to {output_path}...")
df.to_csv(output_path, index=False)

print("\nDone! You can now start the backend.")
