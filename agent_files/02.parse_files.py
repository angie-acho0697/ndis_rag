from pathlib import Path
import docx
import pandas as pd
from tqdm import tqdm
import time
import yaml
from fnmatch import fnmatch


# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
file_patterns = config.get("file_patterns", None)

def parse_files_and_web_page(directory, web_page_file):
    combined_content = []
    print("\nStarting data processing...")

    # Add web page content
    if web_page_file:
        try:
            print("\nProcessing web page content...")
            with open(web_page_file, "r", encoding="utf-8") as f:
                page_content = f.read()
                combined_content.append(f"Web Page Content:\n{page_content}")
            print("✓ Web page content processed successfully")
        except Exception as e:
            print(f"Error reading web page file {web_page_file}: {e}")

    # Get list of files to process
    all_files = list(Path(directory).glob("*"))
    if file_patterns:
        files = [f for f in all_files if any(fnmatch(f.name, pat) for pat in file_patterns)]
    else:
        files = all_files
    print(f"\nFound {len(files)} files to process")

    # Parse CSV and DOCX files
    for file_path in tqdm(files, desc="Processing files", unit="file"):
        if file_path.suffix.lower() == ".csv":
            try:
                # Try different encodings
                encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
                df = None

                for encoding in tqdm(
                    encodings, desc=f"Trying encodings for {file_path.name}", leave=False
                ):
                    try:
                        df = pd.read_csv(
                            file_path,
                            encoding=encoding,
                            low_memory=False,  # Handle mixed types
                            dtype=str,  # Read all columns as string to avoid type issues
                        )
                        break
                    except UnicodeDecodeError:
                        continue

                if df is not None:
                    # Show progress for converting to string
                    with tqdm(
                        total=1, desc=f"Converting {file_path.name} to string", leave=False
                    ) as pbar:
                        content = df.to_string(index=False)
                        combined_content.append(f"File: {file_path.name}\n{content}")
                        pbar.update(1)
                else:
                    print(f"Could not read CSV {file_path} with any of the attempted encodings")

            except Exception as e:
                print(f"Error reading CSV {file_path}: {e}")

        elif file_path.suffix.lower() == ".docx":
            try:
                # Show progress for DOCX processing
                with tqdm(total=1, desc=f"Processing {file_path.name}", leave=False) as pbar:
                    doc = docx.Document(file_path)
                    content = "\n".join(
                        [para.text for para in doc.paragraphs if para.text.strip()]
                    )
                    combined_content.append(f"File: {file_path.name}\n{content}")
                    pbar.update(1)
            except Exception as e:
                print(f"Error reading DOCX {file_path}: {e}")

    print("\nCombining all content...")
    # Show progress for final combination
    with tqdm(total=1, desc="Finalizing combined content") as pbar:
        final_content = "\n\n".join(combined_content)
        pbar.update(1)

    return final_content


if __name__ == "__main__":
    start_time = time.time()

    print("Starting file parsing process...")
    data_dir = "data/raw"
    web_page_file = "data/raw/web_page_content.txt"

    full_context = parse_files_and_web_page(data_dir, web_page_file)

    # Output the combined content to a new file
    output_file = Path("data/processed/combined_content.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("\nSaving combined content...")
    with tqdm(total=1, desc="Writing to file") as pbar:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_context)
        pbar.update(1)

    end_time = time.time()
    print(f"\n✓ Process completed in {end_time - start_time:.2f} seconds")
    print(f"Combined content saved to {output_file}")
