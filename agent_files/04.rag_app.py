import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import re

import streamlit as st
import yaml


# Load config
def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    print("Looking for config at:", config_path.resolve())
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        # Expand environment variables in the Ollama path
        if "ollama_path" in config:
            config["ollama_path"] = os.path.expandvars(config["ollama_path"])
        return config


config = load_config()

st.set_page_config(
    page_title="NDIS Q&A System",
    page_icon=":guardsman:",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

OLLAMA_PATH = config.get("ollama_path")
MODEL_NAME = config.get("model_name", "llama3")
EMBEDDING_MODEL = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = config.get("chunk_size", 3000)
CHUNK_OVERLAP = config.get("chunk_overlap", 100)
FILE_PATH = config.get("file_path")

module_path = os.path.join(os.path.dirname(__file__), "03.qanda_rag.py")
module_name = "qanda_rag_temp"

spec = importlib.util.spec_from_file_location(module_name, module_path)
qanda_rag = importlib.util.module_from_spec(spec)
sys.modules[module_name] = qanda_rag
spec.loader.exec_module(qanda_rag)

Llama3QASystem = qanda_rag.Llama3QASystem


# Function to check if Ollama is installed
def is_ollama_installed() -> bool:
    try:
        result = subprocess.run(
            [OLLAMA_PATH, "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print("Ollama stdout:", result.stdout)
        print("Ollama stderr:", result.stderr)
        print("Ollama returncode:", result.returncode)
        return result.returncode == 0
    except Exception as e:
        print("Ollama check error:", e)
        return False


# Function to check if Llama 3 is available in Ollama
def is_llama3_available() -> bool:
    try:
        result = subprocess.run(
            [OLLAMA_PATH, "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            return "llama3" in result.stdout
        return False
    except Exception:
        return False


# Function to download Llama 3 in the background
def download_llama3_bg(status_placeholder):
    try:
        status_placeholder.text("Downloading Llama 3... This may take several minutes.")
        result = subprocess.run(
            [OLLAMA_PATH, "pull", "llama3"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            status_placeholder.success("Llama 3 downloaded successfully!")
        else:
            status_placeholder.error(f"Failed to download Llama 3: {result.stderr}")
    except Exception as e:
        status_placeholder.error(f"Error downloading model: {str(e)}")


# Function to get Ollama installation instructions
def get_ollama_install_instructions() -> str:
    system = platform.system()
    if system == "Darwin":  # macOS
        return "```\ncurl -fsSL https://ollama.com/install.sh | sh\n```"
    elif system == "Linux":
        return "```\ncurl -fsSL https://ollama.com/install.sh | sh\n```"
    elif system == "Windows":
        return "Download and install Ollama from: https://ollama.com/download/windows"
    else:
        return "Visit https://ollama.com for installation instructions."


def main():
    current_dir = Path(__file__).parent
    logo_path = current_dir.parent / "data" / "external" / "logo.png"

    st.image(str(logo_path), width=150)
    st.title("NDIS Q&A System")
    st.markdown("""
    This application uses Meta's Llama 3 model to answer questions from information on NDIS participants dataset webpage.
    No API keys or payments required - everything runs locally on your machine!
    """)

    # Check if Ollama is installed
    ollama_installed = is_ollama_installed()

    # Check if Llama 3 is available
    llama3_available = is_llama3_available() if ollama_installed else False

    # Sidebar for configuration
    with st.sidebar:
        st.header("System Configuration")

        # Ollama installation status
        if ollama_installed:
            st.success("✅ Ollama is installed")

            # Llama 3 availability status
            if llama3_available:
                st.success("✅ Llama 3 is available")
            else:
                st.warning("⚠️ Llama 3 is not downloaded yet")
                if st.button("Download Llama 3"):
                    # Create a placeholder for the download status
                    download_status = st.empty()
                    # Start download in background thread
                    thread = threading.Thread(target=download_llama3_bg, args=(download_status,))
                    thread.start()

            st.subheader("Embedding Model")
            st.info(EMBEDDING_MODEL)
        else:
            st.error("❌ Ollama is not installed")
            st.markdown("### Installation Instructions:")
            st.markdown(get_ollama_install_instructions())
            st.markdown("After installing, restart this application.")
            # Early return as we can't proceed without Ollama
            st.warning(
                "After installing Ollama, you'll need to download Llama 3 by running: `ollama pull llama3`"
            )
            return

        st.subheader("Document Processing")
        st.info(f"**Chunk Size:** {CHUNK_SIZE}")
        st.info(f"**Chunk Overlap:** {CHUNK_OVERLAP}")

        st.subheader("Index Management")
        index_name = "ndis_participant_index"
        st.markdown(
            f"""
            <div style="background-color:#4B2E5E;padding:12px 16px;border-radius:8px;color:white;font-weight:bold;margin-bottom:8px;">
                {index_name}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("The index will be loaded if it exists, or created and saved automatically.")

    # Initialize session state
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Path to your combined_content.txt file
    index_dir = Path("data/processed") / index_name

    # Warning if Llama 3 is not available
    if not llama3_available and ollama_installed:
        st.warning(
            "⚠️ Llama 3 model is not downloaded yet. Please download it from the sidebar before proceeding."
        )

    # Track last-used settings
    settings = {
        "file_path": FILE_PATH,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "model_name": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
    }

    # Only rebuild if settings have changed or not processed yet
    file_path_obj = Path(FILE_PATH).resolve()
    if (
        not st.session_state.get("qa_system")
        or not st.session_state.get("file_processed")
        or st.session_state.get("last_settings") != settings
    ) and llama3_available:
        if not file_path_obj.exists():
            st.error(f"Knowledge file not found: {file_path_obj}")
        else:
            with st.spinner("Setting up the knowledge base with Llama 3..."):
                qa_system = Llama3QASystem(
                    file_path=str(file_path_obj),
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    model_name=MODEL_NAME,
                    embedding_model=EMBEDDING_MODEL,
                )
                # Try to load the index first
                index_file = index_dir
                if index_file.exists():
                    loaded = qa_system.load_index(index_file)
                    if loaded:
                        qa_system.setup_qa_chain()
                        st.session_state.qa_system = qa_system
                        st.session_state.file_processed = True
                        st.session_state.last_settings = settings
                        st.success(
                            f"Index '{index_name}' loaded successfully! You can now ask questions."
                        )
                    else:
                        st.session_state.qa_system = None
                        st.session_state.file_processed = False
                        st.error(f"Failed to load index '{index_name}'.")
                else:
                    # Build and save the index if it doesn't exist
                    success = qa_system.setup()
                    if success:
                        qa_system.save_index(index_name)
                        st.session_state.qa_system = qa_system
                        st.session_state.file_processed = True
                        st.session_state.last_settings = settings
                        st.success(
                            f"Knowledge base processed and index saved as '{index_name}'! You can now ask questions."
                        )
                    else:
                        st.session_state.qa_system = None
                        st.session_state.file_processed = False
                        st.error(
                            "Error processing the knowledge base. Please check the console for details."
                        )

    # Question asking area
    st.header("Ask Questions")

    if st.session_state.file_processed and st.session_state.qa_system and llama3_available:
        # Chat interface
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

        # Question input
        question = st.chat_input("What would you like to know about the knowledge base?")

        if question:
            # Add question to chat history
            with st.chat_message("user"):
                st.write(question)

            # Display thinking spinner
            with st.chat_message("assistant"):
                with st.spinner("Llama 3 is thinking..."):
                    result = st.session_state.qa_system.answer_question(question)
                    answer = Llama3QASystem.clean_llama_output(result["answer"])
                    st.write(answer)

                    # Show sources in an expander
                    with st.expander("View Sources"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**Source {i + 1}:**")
                            st.text(doc.page_content)

            # Update chat history
            st.session_state.chat_history.append((question, answer))
    else:
        if not llama3_available:
            st.info("Please download Llama 3 from the sidebar first.")
        else:
            st.info("Please wait while the knowledge base is being processed...")

    # Footer
    st.markdown("---")
    st.markdown("""
    **How this Llama 3 Q&A system works:**
    
    1. **Document Processing**: Your document is split into smaller chunks and converted into numerical vectors using the embedding model.
    
    2. **Vector Database**: These vectors are stored in a FAISS index that allows for efficient similarity search.
    
    3. **Question Answering**: When you ask a question:
       - The question is converted to a vector and used to find the most relevant document chunks
       - The relevant chunks and your question are sent to Llama 3
       - Llama 3 generates an answer based on the context provided by those chunks
    
    All processing happens locally on your device - your data never leaves your computer!
    """)

    st.markdown(
        "**Disclaimer**: This is a demo application. For production use, consider using a more robust setup."
    )


if __name__ == "__main__":
    main()
