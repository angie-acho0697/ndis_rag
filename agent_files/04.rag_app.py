# Import the Llama3QASystem class (assumes it's defined in a file named qanda_rag.py)
import importlib.util
from pathlib import Path
import platform
import subprocess
import sys
import threading

import streamlit as st

module_path = "03.qanda_rag.py"
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
            ["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# Function to check if Llama 3 is available in Ollama
def is_llama3_available() -> bool:
    try:
        result = subprocess.run(
            ["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
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
            ["ollama", "pull", "llama3"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
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
    st.set_page_config(
        page_title="Llama 3 Q&A System",
        page_icon="🦙",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🦙 Llama 3 Q&A System")
    st.markdown("""
    This application uses Meta's Llama 3 model to answer questions from your knowledge base.
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

            # Select embedding model
            st.subheader("Embedding Model")
            embedding_options = [
                "sentence-transformers/all-MiniLM-L6-v2",  # Fast, small (~80MB)
                "sentence-transformers/all-mpnet-base-v2",  # Better quality (~420MB)
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual
            ]
            embedding_model = st.selectbox(
                "Select Embedding Model",
                options=embedding_options,
                index=0,
                help="Model used for text embeddings. MiniLM is faster, mpnet is more accurate.",
            )

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

        # Document Processing Settings
        st.subheader("Document Processing")
        chunk_size = st.slider(
            "Chunk Size",
            min_value=500,
            max_value=4000,
            value=1000,
            step=100,
            help="Size of text chunks for processing. Larger chunks provide more context but may reduce precision.",
        )

        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between chunks to maintain context across chunk boundaries.",
        )

        # Index Management
        st.subheader("Index Management")
        index_name = st.text_input("Index Name", value="llama3_qa_index")
        col1, col2 = st.columns(2)
        save_index = col1.button("Save Index")
        load_index = col2.button("Load Index")

    # Initialize session state
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Path to your combined_content.txt file
    combined_file_path = Path("data/processed/combined_content.txt").resolve()

    # Warning if Llama 3 is not available
    if not llama3_available and ollama_installed:
        st.warning(
            "⚠️ Llama 3 model is not downloaded yet. Please download it from the sidebar before proceeding."
        )

    # Process the combined_content.txt file (only once)
    if not st.session_state.file_processed and llama3_available:
        if not combined_file_path.exists():
            st.error(f"Knowledge file not found: {combined_file_path}")
        else:
            with st.spinner("Processing the knowledge base with Llama 3..."):
                try:
                    st.session_state.qa_system = Llama3QASystem(
                        file_path=str(combined_file_path),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        model_name="llama3",  # Explicitly use llama3
                        embedding_model=embedding_model,
                    )
                    success = st.session_state.qa_system.setup()
                    if success:
                        st.session_state.file_processed = True
                        st.success(
                            f"Knowledge base processed successfully with Llama 3! You can now ask questions."
                        )
                    else:
                        st.error(
                            "Error processing the knowledge base. Please check the console for details."
                        )
                except Exception as e:
                    st.error(f"Error initializing the QA system: {str(e)}")

    # Save index if requested
    if save_index and st.session_state.qa_system:
        with st.spinner("Saving index..."):
            success = st.session_state.qa_system.save_index(index_name)
            if success:
                st.sidebar.success(f"Index saved as '{index_name}'")
            else:
                st.sidebar.error("Failed to save index")

    # Load index if requested
    if load_index:
        if not llama3_available:
            st.sidebar.error(
                "Cannot load index: Llama 3 model is not available. Please download it first."
            )
        else:
            index_path = Path(index_name)
            if index_path.exists():
                with st.spinner("Loading index..."):
                    if st.session_state.qa_system is None:
                        # Create a minimal QA system if none exists
                        st.session_state.qa_system = Llama3QASystem(
                            model_name="llama3", embedding_model=embedding_model
                        )

                    success = st.session_state.qa_system.load_index(index_name)
                    if success:
                        st.session_state.qa_system.setup_qa_chain()
                        st.session_state.file_processed = True
                        st.sidebar.success(f"Index '{index_name}' loaded successfully")
                    else:
                        st.sidebar.error(f"Failed to load index '{index_name}'")
            else:
                st.sidebar.error(f"Index '{index_name}' not found")

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
                    answer = result["answer"]
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
