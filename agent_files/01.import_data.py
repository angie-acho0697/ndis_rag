import logging
from pathlib import Path
import re
import time
from urllib.parse import urljoin
import yaml
import chardet

from bs4 import BeautifulSoup
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
file_patterns = config.get("file_patterns", [])

def sanitize_filename(name, extension):
    """Remove invalid characters from filenames and ensure the extension."""
    # Convert to ASCII, replacing non-ASCII characters
    name = name.encode('ascii', 'replace').decode('ascii')
    name = re.sub(r"[^A-Za-z0-9 _.-]", "", name).strip().replace(" ", "_")
    return f"{name}.{extension}"

def matches_pattern(filename, patterns):
    """Check if filename matches any of the patterns."""
    for pattern in patterns:
        # Convert pattern to regex pattern
        regex_pattern = pattern.replace('*', '.*')
        if re.match(regex_pattern, filename, re.IGNORECASE):
            return True
    return False

def ensure_ascii_content(content):
    """Convert content to ASCII, replacing non-ASCII characters."""
    if isinstance(content, bytes):
        # Detect encoding
        detected = chardet.detect(content)
        encoding = detected['encoding'] or 'utf-8'
        content = content.decode(encoding, errors='replace')
    
    # Convert to ASCII
    return content.encode('ascii', 'replace').decode('ascii')

def extract_links_from_text(url, file_types=("CSV", "DOCX")):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract page content and ensure ASCII
    page_text = soup.get_text(strip=True)
    page_text = ensure_ascii_content(page_text)
    logging.info("Extracted web page text.")

    matched_links = []
    for link in soup.find_all("a", href=True):
        link_text = link.get_text(strip=True)
        link_text = ensure_ascii_content(link_text)
        href = link["href"]

        for file_type in file_types:
            if file_type in link_text.upper():
                full_url = urljoin(url, href)
                filename = sanitize_filename(link_text, file_type.lower())
                
                # Only include files that match the patterns
                if file_patterns and not matches_pattern(filename, file_patterns):
                    logging.info(f"Skipping non-matching file: {filename}")
                    continue
                    
                matched_links.append((filename, full_url))
                logging.info(f"Found matching file: {filename}")
                break  # avoid duplicate match if both CSV and DOCX are in link text

    return matched_links, page_text

def download_file(download_url, output_path):
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        content = response.content

        # For CSV files, handle carefully to prevent corruption
        if output_path.suffix.lower() == '.csv':
            # Check if content appears to be Excel file (has PK header)
            if content.startswith(b'PK\x03\x04'):
                logging.warning(f"File {output_path.name} appears to be an Excel file. Converting to CSV...")
                try:
                    import pandas as pd
                    import io
                    
                    # Try reading with openpyxl engine first
                    try:
                        excel_data = pd.read_excel(io.BytesIO(content), engine='openpyxl')
                    except Exception as e1:
                        logging.warning(f"Openpyxl engine failed: {e1}. Trying xlrd engine...")
                        try:
                            # Fall back to xlrd engine
                            excel_data = pd.read_excel(io.BytesIO(content), engine='xlrd')
                        except Exception as e2:
                            logging.error(f"Both Excel engines failed: {e2}")
                            raise
                    
                    # Convert to CSV with proper encoding
                    csv_data = excel_data.to_csv(index=False, encoding='ascii', errors='replace')
                    
                    # Write as ASCII
                    with open(output_path, "w", encoding='ascii') as f:
                        f.write(csv_data)
                    
                    logging.info(f"Successfully converted Excel to CSV: {output_path}")
                except Exception as e:
                    logging.error(f"Error converting Excel to CSV: {e}")
                    # Save original content with .xlsx extension
                    xlsx_path = output_path.with_suffix('.xlsx')
                    with open(xlsx_path, "wb") as f:
                        f.write(content)
                    logging.info(f"Saved original Excel file as: {xlsx_path}")
            else:
                # Try to detect encoding
                detected = chardet.detect(content)
                encoding = detected['encoding'] or 'utf-8'
                
                try:
                    # Read content with detected encoding
                    text_content = content.decode(encoding, errors='replace')
                    # Convert to ASCII
                    ascii_content = ensure_ascii_content(text_content)
                    
                    # Write as ASCII
                    with open(output_path, "w", encoding='ascii') as f:
                        f.write(ascii_content)
                except Exception as e:
                    logging.error(f"Error processing CSV: {e}")
                    # Save original content
                    with open(output_path, "wb") as f:
                        f.write(content)
        else:
            # For other files (like DOCX), write as binary
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        logging.info(f"Downloaded to: {output_path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {download_url}: {e}")
    except Exception as e:
        logging.error(f"Error processing {download_url}: {e}")

def main():
    url = "https://dataresearch.ndis.gov.au/datasets/participant-datasets"
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the download links and the page content
    file_links, page_content = extract_links_from_text(url, file_types=("CSV", "DOCX"))

    # Save the page content to a text file in ASCII
    page_content_file = output_dir / "web_page_content.txt"
    with open(page_content_file, "w", encoding="ascii") as f:
        f.write(page_content)
    logging.info(f"Saved web page content to {page_content_file}")

    logging.info(f"Found {len(file_links)} matching file links.")

    start_time = time.time()  # Start timer
    logging.info("Starting downloads...")

    for filename, file_url in file_links:
        output_file = output_dir / filename
        download_file(file_url, output_file)

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time

    logging.info(f"Downloaded {len(file_links)} files in {elapsed_time:.2f} seconds.")
    logging.info("All downloads completed.")

if __name__ == "__main__":
    main()
