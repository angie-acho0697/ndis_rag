import logging
from pathlib import Path
import re
import time
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def sanitize_filename(name, extension):
    """Remove invalid characters from filenames and ensure the extension."""
    name = re.sub(r"[^A-Za-z0-9 _.-]", "", name).strip().replace(" ", "_")
    return f"{name}.{extension}"


def extract_links_from_text(url, file_types=("CSV", "DOCX")):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract page content
    page_text = soup.get_text(strip=True)  # Get all the text from the page
    logging.info("Extracted web page text.")

    matched_links = []
    for link in soup.find_all("a", href=True):
        link_text = link.get_text(strip=True)
        href = link["href"]

        for file_type in file_types:
            if file_type in link_text.upper():
                full_url = urljoin(url, href)
                filename = sanitize_filename(link_text, file_type.lower())
                matched_links.append((filename, full_url))
                break  # avoid duplicate match if both CSV and DOCX are in link text

    return matched_links, page_text


def download_file(download_url, output_path):
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Downloaded to: {output_path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {download_url}: {e}")


def main():
    url = "https://dataresearch.ndis.gov.au/datasets/participant-datasets"
    output_dir = Path("ndis_agent/data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the download links and the page content
    file_links, page_content = extract_links_from_text(url, file_types=("CSV", "DOCX"))

    # Save the page content to a text file
    page_content_file = output_dir / "web_page_content.txt"
    with open(page_content_file, "w", encoding="utf-8") as f:
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
