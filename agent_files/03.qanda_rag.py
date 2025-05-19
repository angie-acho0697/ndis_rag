from functools import lru_cache
import os
from pathlib import Path
import time
from typing import Any, Dict
from tqdm import tqdm
import re

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS


class Llama3QASystem:
    def __init__(
        self,
        file_path: str = None,
        chunk_size: int = 3000,
        chunk_overlap: int = 100,
        model_name: str = "llama3",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.vectorstore = None
        self.qa_chain = None
        self.model_name = model_name

        t0 = time.time()
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        print(f"Embedding model loaded in {time.time() - t0:.2f} seconds.")

        self._setup_llm()

    def _setup_llm(self):
        t0 = time.time()
        try:
            print(f"Setting up LLM: {self.model_name}")
            self.llm = Ollama(
                model=self.model_name,
                temperature=0,
                num_ctx=4096,
                num_predict=256,
            )
            print(
                f"Successfully loaded {self.model_name} model via Ollama in {time.time() - t0:.2f} seconds."
            )
        except Exception as e:
            print(f"Error setting up LLM: {e}")
            print("Make sure Ollama is installed and running, and the Llama 3 model is downloaded")
            print("You can download Llama 3 using: ollama pull llama3")

    def load_data(self):
        t0 = time.time()
        print(f"Loading data from {self.file_path}...")
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                text = file.read()
        except Exception as e:
            print(f"Error loading file: {e}")
            return False

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            keep_separator=True,
            is_separator_regex=False
        )
        texts = text_splitter.split_text(text)
        self.documents = [Document(page_content=t, metadata={"source": str(i)}) for i, t in enumerate(texts)]
        print(
            f"Data loaded and split into {len(self.documents)} chunks in {time.time() - t0:.2f} seconds."
        )
        return True

    def build_index(self):
        if not self.documents:
            print("No documents loaded. Please load data first.")
            return False

        t0 = time.time()
        print("Building vector index with embedding progress bar...")

        texts = [doc.page_content for doc in self.documents]
        batch_size = 128
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch)
            embeddings.extend(batch_embeddings)

        print(f"Embeddings generated in {time.time() - t0:.2f} seconds.")

        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=self.embedding_model,
            metadatas=[{"source": str(i)} for i in range(len(texts))],
        )
        print(f"FAISS index built in {time.time() - t0:.2f} seconds.")
        print(f"Vector index built successfully in {time.time() - t0:.2f} seconds.")
        return True

    def setup_qa_chain(self):
        t0 = time.time()
        if not self.vectorstore:
            print("Vector store not initialized. Please build index first.")
            return False

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

        # Configure a more sophisticated retriever
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Use Maximum Marginal Relevance for better diversity
            search_kwargs={
                "k": 4,  # Retrieve more documents
                "fetch_k": 20,  # Consider more candidates
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        print(
            f"QA system is ready to answer questions using Llama 3! (setup in {time.time() - t0:.2f} seconds)"
        )
        return True

    @lru_cache(maxsize=128)
    def answer_question(self, question: str) -> Dict[str, Any]:
        t0 = time.time()
        if not self.qa_chain:
            print("QA system not initialized. Run setup first.")
            return {"answer": "System not ready. Please initialize the QA system first."}
        try:
            result = self.qa_chain({"query": question})
            print(f"Answered question in {time.time() - t0:.2f} seconds.")
            return {
                "answer": result.get("result", "No answer found."),
                "source_documents": result.get("source_documents", []),
            }
        except Exception as e:
            print(f"Error answering question: {e}")
            return {"answer": f"Error processing question: {str(e)}"}

    def setup(self):
        t0 = time.time()
        if self.load_data() and self.build_index() and self.setup_qa_chain():
            print(
                f"Setup complete! You can now ask questions using Llama 3. (Total setup time: {time.time() - t0:.2f} seconds)"
            )
            return True
        return False

    def save_index(self, directory: str = "qa_index"):
        # Absolute path to your desired directory
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "processed")
        )
        save_path = os.path.join(base_dir, directory)
        if not self.vectorstore:
            print("No index to save. Please build index first.")
            return False
        os.makedirs(save_path, exist_ok=True)
        self.vectorstore.save_local(save_path)
        print(f"Index saved to {save_path}")
        return True

    def load_index(self, directory: str = "qa_index"):
        # Absolute path to your desired directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        load_path = os.path.join(base_dir, directory)
        if not os.path.exists(load_path):
            print(f"Index directory {load_path} not found.")
            return False

        try:
            self.vectorstore = FAISS.load_local(
                load_path, self.embedding_model, allow_dangerous_deserialization=True
            )
            print(f"Index loaded from {load_path}")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def clean_llama_output(answer: str) -> str:
        # Remove leading special tokens and whitespace
        return re.sub(r"^(<s>)?\\[.*?\\] *<<SYS>> *", "", answer).strip()
