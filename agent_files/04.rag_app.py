import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys

import streamlit as st
import yaml

# Dynamically import get_query_engine from 05.llamaindex_rag.py
rag_module_path = str(Path(__file__).parent / "05.llamaindex_rag.py")
spec = importlib.util.spec_from_file_location("llamaindex_rag", rag_module_path)
llamaindex_rag = importlib.util.module_from_spec(spec)
sys.modules["llamaindex_rag"] = llamaindex_rag
spec.loader.exec_module(llamaindex_rag)
get_query_engine = llamaindex_rag.get_query_engine

# Suppress Streamlit watcher warning
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


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

OLLAMA_PATH = config.get("ollama_path")
MODEL_NAME = config.get("model_name", "mistral")
EMBEDDING_MODEL = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = config.get("chunk_size", 3000)
CHUNK_OVERLAP = config.get("chunk_overlap", 100)


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


def get_ollama_install_instructions() -> str:
    system = platform.system()
    if system == "Darwin":  # macOS
        return """
        curl -fsSL https://ollama.com/install.sh | sh
        """
    elif system == "Linux":
        return """
        curl -fsSL https://ollama.com/install.sh | sh
        """
    elif system == "Windows":
        return "Download and install Ollama from: https://ollama.com/download/windows"
    else:
        return "Visit https://ollama.com for installation instructions."


# --- Use the RAG pipeline from 05.llamaindex_rag.py ---
@st.cache_resource(show_spinner=True)
def get_cached_query_engine():
    return get_query_engine()


# --- Streamlit UI ---
def main():
    current_dir = Path(__file__).parent
    logo_path = current_dir.parent / "data" / "external" / "logo.png"

    st.image(str(logo_path), width=150)
    st.title("NDIS Q&A System")

    # Get the actual model name from config
    model_name = config.get("model_name")
    st.markdown(f"""
    This application uses {model_name} (via Ollama) to answer questions from information on NDIS participants dataset webpage.
    No API keys or payments required - everything runs locally on your machine!
    """)

    # Check if Ollama is installed
    ollama_installed = is_ollama_installed()

    # Sidebar for configuration
    with st.sidebar:
        st.header("System Configuration")
        if ollama_installed:
            st.success("✅ Ollama is installed")
        else:
            st.error("❌ Ollama is not installed!")
            st.markdown(get_ollama_install_instructions())

        # Model status (from config)
        st.subheader("LLM Model")
        st.success(f"✅ Using {model_name}")

        # Use custom styling for consistent colors
        st.markdown(
            """
        <style>
        /* Blue styling for LLM Model section */
        .llm-info .stInfo {
            background-color: #E8F0FE;
            border: 1px solid #4285F4;
            color: #1F1F1F;
        }
        /* Grey styling for Embedding and Document sections */
        .grey-info .stInfo {
            background-color: #F5F5F5;
            border: 1px solid #9E9E9E;
            color: #1F1F1F;
        }
        /* Pink styling for Search Configuration section */
        .pink-info .stInfo {
            background-color: #FCE4EC;
            border: 1px solid #E91E63;
            color: #1F1F1F;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # LLM Model section with blue styling
        with st.container():
            st.markdown('<div class="llm-info">', unsafe_allow_html=True)
            st.info(f"Temperature: {config.get('llm_temperature', 0)}")
            st.info(f"Top-p: {config.get('llm_top_p', 0.8)}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Embedding Model")
        with st.container():
            st.markdown('<div class="grey-info">', unsafe_allow_html=True)
            st.info(config.get("embedding_model", "Not set"))
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Document Processing")
        with st.container():
            st.markdown('<div class="grey-info">', unsafe_allow_html=True)
            st.info(f"Chunk Size: {config.get('chunk_size', 'Not set')}")
            st.info(f"Chunk Overlap: {config.get('chunk_overlap', 'Not set')}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Search Configuration")
        with st.container():
            st.markdown('<div class="pink-info">', unsafe_allow_html=True)
            st.info("Using hybrid search (FAISS + BM25)")
            st.info("Top-k retrieval: 10 documents")
            st.info("Alpha (dense/sparse balance): 0.5")
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Index Management")
        st.markdown(
            """
            <div style='background-color: #4B2E5E; color: white; padding: 12px; border-radius: 8px; font-weight: bold; text-align: center;'>
                faiss_index
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "The index will be loaded from data/processed/faiss_index if it exists, or created and saved automatically."
        )

    st.header("Ask a Question")
    query_engine = get_cached_query_engine()
    user_question = st.text_area("Enter your question:", height=80)
    if st.button("Get Answer", type="primary") and user_question.strip():
        with st.spinner("Thinking..."):
            try:
                response = query_engine.query(user_question)
                st.markdown("**Answer:**")

                answer = ""
                if hasattr(response, "response_gen"):
                    answer_placeholder = st.empty()
                    for token in response.response_gen:
                        answer += token
                        answer_placeholder.markdown(answer + "▌")  # Typing effect
                    answer_placeholder.markdown(answer)  # Final answer
                elif hasattr(response, "response"):
                    st.markdown(response.response)
                else:
                    st.markdown(str(response))

                # Show source documents if available
                if hasattr(response, "source_nodes"):
                    with st.expander("View Source Documents"):
                        for i, node in enumerate(response.source_nodes, 1):
                            st.markdown(f"**Source {i}:**")
                            try:
                                st.markdown(
                                    f"**Source file:** {node.node.metadata.get('source', 'N/A')}"
                                )
                            except Exception as e:
                                st.markdown(f"Error accessing source file: {e}")
                                st.markdown(f"Metadata keys: {list(node.node.metadata.keys())}")
                            st.markdown(node.node.text)
                            st.markdown("---")

            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.markdown("**Debug Error Info:**")
                st.markdown(f"- Error type: {type(e)}")
                st.markdown(f"- Error message: {str(e)}")


if __name__ == "__main__":
    main()
