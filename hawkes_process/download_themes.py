import requests

# The official URL for the GDELT V2 Master Theme Lookup
url = "http://data.gdeltproject.org/api/v2/guides/LOOKUP-GKGTHEMES.TXT"

# The name of the file as it will be saved on your computer
local_filename = "LOOKUP-GKGTHEMES.TXT"

print(f"Connecting to GDELT servers to download {local_filename}...")

try:
    # stream=True ensures we don't overload your RAM
    with requests.get(url, stream=True) as response:
        response.raise_for_status() # This checks if the URL is valid/accessible
        
        # Open a local file in write-binary mode
        with open(local_filename, 'wb') as file:
            # Download and write the file in 8KB chunks
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                
    print("Download completed successfully! The file is saved in your current directory.")

except requests.exceptions.RequestException as e:
    print(f"An error occurred during the download: {e}")