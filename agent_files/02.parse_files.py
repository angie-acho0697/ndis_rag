from pathlib import Path
import docx
import pandas as pd


def parse_files_and_web_page(directory, web_page_file):
    combined_content = []

    # Add web page content
    if web_page_file:
        try:
            with open(web_page_file, "r", encoding="utf-8") as f:
                page_content = f.read()
                combined_content.append(f"Web Page Content:\n{page_content}")
        except Exception as e:
            print(f"Error reading web page file {web_page_file}: {e}")

    # Parse CSV and DOCX files
    for file_path in Path(directory).glob("*"):
        if file_path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(file_path)
                content = df.to_string(index=False)
                combined_content.append(f"File: {file_path.name}\n{content}")
            except Exception as e:
                print(f"Error reading CSV {file_path}: {e}")
        elif file_path.suffix.lower() == ".docx":
            try:
                doc = docx.Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                combined_content.append(f"File: {file_path.name}\n{content}")
            except Exception as e:
                print(f"Error reading DOCX {file_path}: {e}")

    return "\n\n".join(combined_content)


if __name__ == "__main__":
    data_dir = "data/raw"
    web_page_file = "data/raw/web_page_content.txt"  # Path to the web page content file
    full_context = parse_files_and_web_page(data_dir, web_page_file)

    # Output the combined content to a new file
    output_file = Path("data/processed/combined_content.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_context)

    print(f"Combined content saved to {output_file}")
