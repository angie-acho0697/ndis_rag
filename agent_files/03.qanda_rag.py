import os
from typing import List, Dict, Any
import numpy as np
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from pathlib import Path


class Llama3QASystem:
    def __init__(
        self,
        file_path: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "llama3",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the QA system with Llama 3 as the default model

        Args:
            file_path: Path to the text file containing knowledge
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks to maintain context
            model_name: Name of the Ollama model to use (defaults to llama3)
            embedding_model: HuggingFace embedding model to use
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.vectorstore = None
        self.qa_chain = None

        # Default to llama3, but allow flexibility to use other models through the parameter
        self.model_name = model_name

        # Set up the embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)

        # Set up the LLM
        self._setup_llm()

    def _setup_llm(self):
        """Set up the Llama 3 language model"""
        try:
            print(f"Setting up LLM: {self.model_name}")

            # Connect to Ollama for the Llama 3 model
            self.llm = Ollama(
                model=self.model_name,
                temperature=0,  # Low temperature for more factual responses
                num_ctx=4096,  # Context window size
                num_predict=2048,  # Maximum tokens to generate
            )
            print(f"Successfully loaded {self.model_name} model via Ollama")

        except Exception as e:
            print(f"Error setting up LLM: {e}")
            print("Make sure Ollama is installed and running, and the Llama 3 model is downloaded")
            print("You can download Llama 3 using: ollama pull llama3")

    def load_data(self):
        """Load and preprocess the text data"""
        print(f"Loading data from {self.file_path}...")

        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                text = file.read()
        except Exception as e:
            print(f"Error loading file: {e}")
            return False

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len
        )

        # Create document objects
        texts = text_splitter.split_text(text)
        self.documents = [Document(page_content=t) for t in texts]
        print(f"Data loaded and split into {len(self.documents)} chunks.")
        return True

    def build_index(self):
        """Build a vector index for efficient retrieval"""
        if not self.documents:
            print("No documents loaded. Please load data first.")
            return False

        print("Building vector index...")
        self.vectorstore = FAISS.from_documents(self.documents, self.embedding_model)
        print("Vector index built successfully.")
        return True

    def setup_qa_chain(self):
        """Set up the question answering chain with Llama 3 optimized prompt"""
        if not self.vectorstore:
            print("Vector store not initialized. Please build index first.")
            return False

        # Create a prompt template specifically optimized for Llama 3
        # Llama 3 uses the '<s>[INST]' and '[/INST]' tags for instruction formatting
        prompt_template = """<s>[INST] <<SYS>>
You are a helpful AI assistant built for the National Disability Insurance Scheme which is an Australian government initiative that provides funding and support to individuals with permanent and significant disabilities.
You answers questions based on the provided context.
Your goal is to give accurate, factual information by carefully analyzing the context below.
If the information to answer the question is not in the context, admit that you don't know rather than making up an answer.
If you need to supplement the context with your own knowledge, do so only if it is relevant and factual.
You are not allowed to make up information or provide opinions.
When you use information from the context, reference the specific part you're using.
<</SYS>>

Context:
{context}

Question: {question}

Answer the question thoroughly and accurately based only on the provided context. [/INST]"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Create retrieval QA chain optimized for Llama 3
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" means putting all retrieved docs into the prompt
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 10
                },  # Retrieve top 10 chunks - Llama 3 can handle more context
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

        print("QA system is ready to answer questions using Llama 3!")
        return True

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the Llama 3 QA chain

        Args:
            question: The question to answer

        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            print("QA system not initialized. Run setup first.")
            return {"answer": "System not ready. Please initialize the QA system first."}

        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result.get("result", "No answer found."),
                "source_documents": result.get("source_documents", []),
            }
        except Exception as e:
            print(f"Error answering question: {e}")
            return {"answer": f"Error processing question: {str(e)}"}

    def setup(self):
        """Run the complete setup process"""
        if self.load_data() and self.build_index() and self.setup_qa_chain():
            print("Setup complete! You can now ask questions using Llama 3.")
            return True
        return False

    def save_index(self, directory: str = "qa_index"):
        """Save the vectorstore index for later use"""
        if not self.vectorstore:
            print("No index to save. Please build index first.")
            return False

        os.makedirs(directory, exist_ok=True)
        self.vectorstore.save_local(directory)
        print(f"Index saved to {directory}")
        return True

    def load_index(self, directory: str = "qa_index"):
        """Load a previously saved index"""
        if not os.path.exists(directory):
            print(f"Index directory {directory} not found.")
            return False

        try:
            self.vectorstore = FAISS.load_local(directory, self.embedding_model)
            print(f"Index loaded from {directory}")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False


# Installation guide for Ollama and Llama 3
"""
# To use this system with Llama 3, follow these steps:

## Step 1: Install Ollama
# For macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh

# For Windows:
# Download from https://ollama.com/download/windows

## Step 2: Download Llama 3 model
# Open a terminal/command prompt and run:
ollama pull llama3

# For better quality (if you have enough RAM/GPU memory, about 16GB+):
ollama pull llama3:70b

## Step 3: Run this script
# Make sure Ollama is running in the background
"""

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Llama 3 Q&A system for text files")
    parser.add_argument("--file", type=str, required=True, help="Path to the text file to load")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")

    args = parser.parse_args()
    args.file = str(Path("data/processed/combined_content.txt").resolve())

    # conda install -c conda-forge sentence-transformers
    # Initialize QA system with Llama 3
    qa_system = Llama3QASystem(file_path=args.file)

    # Set up the system
    qa_system.setup()

    # Interactive mode
    if args.interactive:
        print("\nEnter your questions. Type 'exit' to quit.")
        while True:
            question = input("\nQuestion: ")
            if question.lower() in ["exit", "quit", "q"]:
                break

            result = qa_system.answer_question(question)
            print("\nAnswer:", result["answer"])

            # Optionally show source documents
            if input("\nShow source documents? (y/n): ").lower() == "y":
                for i, doc in enumerate(result["source_documents"]):
                    print(f"\nSource {i + 1}:")
                    print(
                        doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content
                    )

    print("Llama 3 QA system session ended.")
