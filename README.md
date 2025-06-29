# NDIS Q&A RAG System

[![CCDS Project template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-00A98F?logo=facebook)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-7C3AED?logo=ollama)](https://ollama.ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FF6B6B?logo=huggingface)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-00A67E?logo=langchain)](https://python.langchain.com)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)](https://www.python.org)

A Q&A system built to answer questions about NDIS (National Disability Insurance Scheme) participants using Meta's Llama 3 model. This system processes NDIS data and aims to provide accurate, context-aware answers to questions about NDIS participants.

## 🌟 Features

- **Local Processing**: All processing happens on your machine - no API costs or data privacy concerns
- **Context Aware RAG System**: Uses Retrieval-Augmented Generation (RAG) for accurate, context-aware answers
- **Interactive UI**: Beautiful Streamlit interface for easy interaction
- **Document Processing**: Handles multiple file formats (CSV, DOCX) and web content
- **Vector Search**: Similarity search using FAISS
- **Customizable**: Adjustable chunk sizes and overlap for optimal performance

## 🚀 Quick Start (Windows)

1. **Clone the repository**


```bash
git clone [repository-url]
cd ndis_rag
cd ndis_rag
```

2. **Create and activate virtual environment**


```bash
python -m venv ndis_qa_env
ndis_qa_env\Scripts\activate
```

3. **Install dependencies**


```bash
pip install -r requirements.txt
```

4. **Install Ollama**

Download and install Ollama from [Ollama Windows](https://ollama.com/download/windows)

5. **Download Llama 3**


```bash
ollama pull llama3
```

6. **Run the data prep**

```bash
python agent_files/01.import_data.py
python agent_files/02.parse_file.py

```

7. **Run the application, which also runs the embedding and indexing step**

```bash
streamlit run agent_files/04.rag_app.py
```

## 🖥️ Example System Specifications

The following are the specifications of the system used for development and testing of this project:

| Component         | Details                                                      |
|-------------------|--------------------------------------------------------------|
| **OS**            | Windows 11 Pro 64-bit (10.0, Build 26100)                    |
| **Processor**     | 11th Gen Intel(R) Core(TM) i7-11390H @ 3.40GHz (8 CPUs)      |
| **Memory**        | 16 GB RAM                                                    |

> **Note:** While the application should run on any modern Windows 10/11 machine, performance may vary depending on your hardware (especially RAM and CPU). For best results, use a system with at least 16GB RAM and a recent multi-core processor.

## 📁 Project Structure

```
├── LICENSE            <- Open-source license
├── Makefile          <- Makefile with convenience commands
├── README.md         <- Project documentation
├── config.yaml       <- Configuration settings
├── requirements.txt  <- Python dependencies
├── pyproject.toml    <- Project metadata and tool configuration
├── data/            <- Data directory
│   ├── external/    <- Third-party data
│   ├── interim/     <- Intermediate data
│   ├── processed/   <- Final datasets
│   └── raw/        <- Original data
└── agent_files/     <- Core application files
    ├── 01.import_data.py    <- Data collection script
    ├── 02.parse_files.py    <- Data processing script
    ├── 03.llamaindex_rag.py      <- RAG system implementation
    └── 04.rag_app.py        <- Streamlit application
```

## 🔧 How It Works

1. **Data Collection** (`agent_files/01.import_data.py`)
   - Scrapes NDIS participant data from the official website
   - Downloads relevant CSV and DOCX files defined under config file
   - Extracts and saves web page content

2. **Data Processing** (`agent_files/02.parse_files.py`)
   - Combines content from multiple sources
   - Processes different file formats
   - Creates a unified knowledge base

3. **Q&A System** (`agent_files/03.llamaindex_rag.py`, `agent_files/04.rag_app.py`)
   - Splits documents into manageable chunks
   - Creates vector embeddings using sentence transformers
   - Builds a FAISS index for efficient similarity search
   - Uses Llama 3 for generating accurate answers
   - Provides a user-friendly Streamlit interface

## ⚙️ Configuration

The system can be configured through `config.yaml`. Note that I have created the system on a subset of data. You can specify the files you want the system to be based on in here under the `file_patterns` parameter. If you want to create the rag system on all datasets, you will need to do further fine tuning with a much more powerful machine, and remove or comment out the `file_patterns` parameter.

Also due to the same machine constraint, I have reduced my search space to look at NSW Central Coast only. Uou can adjust this using the `state_filter` and 'srvc_dstrct_filter` under `config.yaml`.


```yaml
ollama_path: "path/to/ollama"
model_name: "llama3"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
chunk_size: 3000
chunk_overlap: 100
file_path: "path/to/your/data"
# (Optional) Patterns for files to include in processing.
# If omitted or left empty, all files in the directory will be processed.
file_patterns:
  - "Participant_numbers_and_plan_budgets_data_*"
  - "Participant_numbers_and_plan_budgets_data_rules_DOCX*"
state_filter: ["NSW"]
srvc_dstrct_filter: ["Central Coast"]  
```

## 🎯 Usage

1. Ensure Ollama is running in the background
2. Launch the application using `streamlit run agent_files/04.rag_app.py`
3. Wait for the system to process the knowledge base
4. Type your questions in the chat interface
5. View answers with source references

## ⚡ Performance Considerations

- For better performance, use a machine with GPU
- Adjust chunk sizes and overlap based on your needs
- For larger datasets, consider using the 70B parameter model:
  ```bash
  ollama pull llama3:70b
  ```

## 🔍 Troubleshooting

- **CUDA/GPU Errors**: Try forcing CPU usage
- **Slow Performance**: Reduce chunk sizes or use a smaller embedding model
- **Incomplete Answers**: Increase the number of retrieved chunks
- **Ollama Location**: Use `where ollama.exe` (Windows) to find the correct path

## ⚠️ Important Disclaimer

**This project is for learning and experimentation purposes only. It is not affiliated with, endorsed by, or connected to the National Disability Insurance Scheme (NDIS) in any way.**

### Data Usage and Licensing


The NDIS data used in this project is publicly available and is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. This license allows:


- Sharing (copying and redistributing the material)
- Adapting (remixing, transforming, and building upon the material)
- Using for educational, research, and non-commercial purposes

## 📝 Important Notes


- This is a demonstration project showcasing the use of RAG (Retrieval-Augmented Generation) systems with public data
- The answers provided by this system should not be considered official NDIS advice or information
- For accurate and official NDIS information, please visit [the official NDIS website](https://www.ndis.gov.au)
- **Performance Consideration**: The local RAG implementation used in this project may exhibit slower response times compared to direct API calls to language models. Users should be aware of this performance characteristic when utilising the system
- **Testing Configuration**: For initial testing and development purposes, users may substitute the full dataset created from the 02.parse_files.py file with a smaller sample dataset. This should contain a subset of the parsed content and is recommended for preliminary testing and validation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025, Angelica Chowdhury

The MIT License is a permissive license that is short and to the point. It lets people do anything they want with your code as long as they provide attribution back to you and don't hold you liable.

Key permissions under MIT License:


- Commercial use
- Modification
- Distribution
- Private use

For more information about the MIT License, visit [choosealicense.com/licenses/mit/](https://choosealicense.com/licenses/mit/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
