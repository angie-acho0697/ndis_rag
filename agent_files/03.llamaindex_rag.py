import json
import os
from pathlib import Path
import time

import faiss
from httpx import ReadTimeout
from llama_index.core import (
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.schema import Document as LIDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import requests.exceptions
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm
import yaml


def get_config_value(config, key, default=None):
    return config[key] if key in config else default


def chunk_text(text, chunk_size=3000, overlap=100):
    """Chunk long text for better retrieval while preserving semantic boundaries."""
    # First split into major sections (tables, paragraphs, etc.)
    sections = []
    current_section = []
    lines = text.split("\n")

    for line in lines:
        # Check if line is a table header or separator
        if "|" in line and ("---" in line or all(c in "|-" for c in line.strip())):
            if current_section:
                sections.append("\n".join(current_section))
                current_section = []
            sections.append(line)  # Keep table structure intact
        # Check if line is a table row
        elif "|" in line:
            if current_section and "|" not in current_section[0]:
                sections.append("\n".join(current_section))
                current_section = []
            current_section.append(line)
        # Regular text
        else:
            if current_section and "|" in current_section[0]:
                sections.append("\n".join(current_section))
                current_section = []
            current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section))

    # Now chunk the sections while preserving table structure
    chunks = []
    current_chunk = []
    current_size = 0

    for section in sections:
        section_words = section.split()
        section_size = len(section_words)

        # If section is a table, try to keep it intact
        if "|" in section:
            if current_size + section_size > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(section)
            current_size += section_size
        else:
            # For regular text, use sliding window with overlap
            if current_size + section_size > chunk_size:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                # Start new chunk with overlap
                overlap_words = current_chunk[-1].split()[-overlap:] if current_chunk else []
                current_chunk = [" ".join(overlap_words)]
                current_size = len(overlap_words)

            current_chunk.append(section)
            current_size += section_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def build_index_with_progress(all_nodes, storage_context):
    start_time = time.time()
    index = None
    batch_size = 100  # Adjust as needed
    batched_nodes = [all_nodes[i : i + batch_size] for i in range(0, len(all_nodes), batch_size)]
    for batch_nodes in tqdm(batched_nodes, desc="Embedding", unit="node"):
        if index is None:
            index = VectorStoreIndex.from_documents(batch_nodes, storage_context=storage_context)
        else:
            # Add to index using insert_nodes if available, otherwise rebuild
            if hasattr(index, "insert_nodes"):
                index.insert_nodes(batch_nodes)
            else:
                # Fallback: rebuild index from all nodes
                index = VectorStoreIndex.from_documents(all_nodes, storage_context=storage_context)
                break
    elapsed = time.time() - start_time
    print(f"Index build time: {elapsed:.2f} seconds")
    return index, elapsed


def hybrid_search(index, query, k=10, alpha=0.5):
    """Perform hybrid search combining dense and sparse retrieval."""
    # Dense retrieval using FAISS
    dense_results = index.as_retriever(search_type="similarity", search_kwargs={"k": k}).retrieve(
        query
    )

    # Sparse retrieval using BM25
    texts = [node.node.text for node in dense_results]
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    tokenized_query = query.split()
    sparse_scores = bm25.get_scores(tokenized_query)

    # Combine scores
    combined_results = []
    for i, node in enumerate(dense_results):
        dense_score = node.score if hasattr(node, "score") else 0
        sparse_score = sparse_scores[i]
        combined_score = alpha * dense_score + (1 - alpha) * sparse_score
        combined_results.append((node, combined_score))

    # Sort by combined score
    combined_results.sort(key=lambda x: x[1], reverse=True)
    return [node for node, _ in combined_results[:k]]


def get_query_engine(progress_callback=None):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    embedding_model = get_config_value(
        config, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    model_name = get_config_value(config, "model_name", "mistral")
    get_config_value(config, "ollama_path", None)
    llm_temperature = get_config_value(config, "llm_temperature", 0)
    llm_top_p = get_config_value(config, "llm_top_p", 0.8)
    chunk_size = get_config_value(config, "chunk_size", 2000)
    chunk_overlap = get_config_value(config, "chunk_overlap", 400)
    state_filter = get_config_value(config, "state_filter", [])
    srvc_dstrct_filter = get_config_value(config, "srvc_dstrct_filter", [])
    raw_max_nodes = get_config_value(config, "max_nodes", 10000)
    try:
        max_nodes = int(raw_max_nodes)
    except (ValueError, TypeError):
        max_nodes = 10000

    print(f"Using embedding model: {embedding_model}")
    print(f"Using LLM: {model_name} (temperature={llm_temperature}, top_p={llm_top_p})")
    print(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")
    print(f"Maximum nodes to process: {max_nodes}")
    if state_filter and srvc_dstrct_filter:
        print(
            f"Filtering CSV rows by StateCd in: {state_filter} and SrvcDstrctNm in: {srvc_dstrct_filter}"
        )
    else:
        print("Indexing all rows (no StateCd or SrvcDstrctNm filter)")

    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    Settings.embed_model = embed_model

    system_prompt = """You are a specialized AI assistant for the National Disability Insurance Scheme (NDIS), an Australian government initiative that provides funding and support to individuals with permanent and significant disabilities.

Your primary goal is to provide accurate, factual responses based on the provided context. Follow these guidelines:

1. CRITICAL: NEVER HALLUCINATE OR MAKE UP INFORMATION
   - If the exact answer is not in the provided context, say "The answer is not available in the provided context"
   - Do not perform calculations unless explicitly requested and the raw data is available
   - Do not assume or infer values that are not directly stated in the context

2. Data Analysis:
   - When analyzing tables, first identify the relevant columns and rows based on the question's criteria (date, disability group, age band, state, region, etc.)
   - For numerical questions, check if the value is already present in the table before performing calculations
   - Show your work by explaining which rows and columns were used
   - If a calculation is needed, break it down step by step

3. Context Usage:
   - Use both data tables and definition tables when available
   - Data tables provide the actual values and statistics
   - Definition tables explain the meaning of terms and columns
   - If the answer requires both, explain the data first, then clarify any terms

4. Response Structure:
   - Start with a direct answer to the question
   - If needed, provide relevant context or definitions
   - For complex answers, use bullet points or sections
   - Include specific numbers and statistics when available

5. Limitations:
   - Only use information from the provided context
   - If information is not in the context, say "The answer is not available in the provided context"
   - Do not make assumptions or provide speculative information
   - Do not include general knowledge about NDIS unless it's in the context

6. Special Cases:
   - For questions about trends or changes over time, compare relevant time periods
   - For questions about specific regions or states, focus on the relevant geographical data
   - For questions about disability groups, ensure you're using the correct terminology from the context

7. Definitions and Calculations:
   - If the context contains a definition for a metric (such as "average annualised committed support budget"), use that definition to answer the question.
   - Do NOT perform your own calculation or make up a definition if a definition is present in the context.
   - Only perform calculations if the value is not explicitly available and no definition is provided in the context.

Remember: Your goal is to help users understand NDIS data accurately and clearly. Always ground your answers in the provided context. If you cannot find the exact information requested, be honest about it."""

    # Define the QA templates
    qa_template = PromptTemplate("""Context information is below.
---------------------
{context_str}
---------------------
Given the context information, please answer the question: {query_str}

IMPORTANT INSTRUCTIONS:
- Look carefully through ALL the provided context for the exact information requested
- Do NOT perform calculations unless the raw data is explicitly available and you are asked to calculate
- If the exact answer is not found in the context, say "The answer is not available in the provided context"
- Be specific about what you found or did not find in the context

Answer:""")

    refine_template = PromptTemplate("""The original question is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
------------
{context_msg}
------------
Given the new context, refine the original answer to better answer the question.
If the context isn't useful, return the original answer.
Refined Answer:""")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ReadTimeout, requests.exceptions.RequestException)),
    )
    def create_ollama_llm():
        return Ollama(
            model=model_name,
            temperature=llm_temperature,
            top_p=llm_top_p,
            request_timeout=600,
            context_window=4096,
            num_ctx=4096,
            system_prompt=system_prompt,
        )

    try:
        llm = create_ollama_llm()
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        print("Please ensure Ollama is running and the model is downloaded")
        raise

    # File paths from config
    csv_files = list(Path("data/processed").glob("*.csv"))
    definitions_path = Path(config["definition_file"]["path"])
    webpage_path = Path(config["web_content_file"]["path"])

    persist_dir = Path("data/processed/faiss_index")
    faiss_index_path = os.path.join(persist_dir, "index.faiss")
    docstore_path = os.path.join(persist_dir, "docstore.json")
    index = None
    all_nodes = None

    # --- Try to load existing index ---
    if os.path.exists(faiss_index_path) and os.path.exists(docstore_path):
        print("Loading existing FAISS index and docstore...")
        faiss_index = faiss.read_index(faiss_index_path)
        with open(docstore_path, "r", encoding="utf-8") as f:
            docstore_data = json.load(f)
        # Reconstruct Document objects (assuming your Document class can be constructed from a dict)
        all_nodes = [LIDocument(**d) for d in docstore_data]
        print(f"Loaded {len(all_nodes)} nodes from docstore")
        print(f"FAISS index has {faiss_index.ntotal} vectors")
        assert len(all_nodes) == faiss_index.ntotal, (
            "Docstore and FAISS index are out of sync! Delete both and rebuild."
        )
        faiss_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=faiss_store)
        index = VectorStoreIndex.from_documents(all_nodes, storage_context=storage_context)
        print("Successfully loaded FAISS index and docstore.")
    else:
        print("Creating new FAISS index...")
        print("Loading CSV files...")
        if progress_callback:
            progress_callback(1, 4, "Loading CSV files...")
        print("CSV files found:")
        for csv_file in csv_files:
            print(f" - {csv_file}")
        data_nodes = []
        for csv_file in tqdm(csv_files, desc="CSV files"):
            try:
                df = pd.read_csv(csv_file, dtype=str)
                if state_filter and "StateCd" in df.columns:
                    df = df[df["StateCd"].isin(state_filter)]
                if srvc_dstrct_filter and "SrvcDstrctNm" in df.columns:
                    df = df[df["SrvcDstrctNm"].isin(srvc_dstrct_filter)]
                for _, row in df.iterrows():
                    metadata = {col: str(row[col]) for col in df.columns}
                    row_text_parts = []
                    for col, val in row.items():
                        if pd.notna(val) and str(val).strip():
                            row_text_parts.append(f"{col.replace('_', ' ')} is {val}")
                    row_text = ". ".join(row_text_parts)
                    data_nodes.append(
                        LIDocument(
                            text=row_text,
                            metadata={"type": "data", "source": csv_file.name, **metadata},
                        )
                    )
            except Exception as e:
                print(f"Warning: Could not load {csv_file}: {e}")
        print("Loading definitions...")
        if progress_callback:
            progress_callback(2, 4, "Loading definitions...")
        definitions_nodes = []
        if definitions_path.exists():
            with open(definitions_path, "r", encoding="utf-8") as f:
                content = f.read()
                chunks = chunk_text(content, chunk_size, chunk_overlap)
                chunks = chunks[: max_nodes // 3]
                for chunk in tqdm(chunks, desc="Definitions chunks"):
                    definitions_nodes.append(
                        LIDocument(
                            text=chunk,
                            metadata={"type": "definition", "source": definitions_path.name},
                        )
                    )
        print("Loading web page content...")
        if progress_callback:
            progress_callback(3, 4, "Loading web page content...")
        webpage_nodes = []
        if webpage_path.exists():
            with open(webpage_path, "r", encoding="utf-8") as f:
                content = f.read()
                chunks = chunk_text(content, chunk_size, chunk_overlap)
                chunks = chunks[: max_nodes // 3]
                for chunk in tqdm(chunks, desc="Webpage chunks"):
                    webpage_nodes.append(
                        LIDocument(
                            text=chunk, metadata={"type": "webpage", "source": webpage_path.name}
                        )
                    )
        all_nodes = data_nodes + definitions_nodes + webpage_nodes
        print(f"Total nodes to embed: {len(all_nodes)}")
        print("Generating embeddings for all nodes...")
        embeddings = [embed_model.get_text_embedding(node.text) for node in all_nodes]
        dimension = len(embeddings[0])
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(embeddings).astype("float32"))
        print(
            f"Saving {len(all_nodes)} nodes to docstore and {faiss_index.ntotal} vectors to FAISS index"
        )
        assert len(all_nodes) == faiss_index.ntotal, (
            "Docstore and FAISS index are out of sync after build!"
        )
        # Save the FAISS index
        os.makedirs(persist_dir, exist_ok=True)
        faiss.write_index(faiss_index, faiss_index_path)
        # Save docstore
        with open(docstore_path, "w", encoding="utf-8") as f:
            json.dump([node.dict() for node in all_nodes], f)
        faiss_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=faiss_store)
        index = VectorStoreIndex.from_documents(all_nodes, storage_context=storage_context)
        print(f"FAISS index and docstore persisted to {persist_dir}")

    # Create query engine with hybrid search
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ReadTimeout, requests.exceptions.RequestException)),
    )
    def create_query_engine():
        return index.as_query_engine(
            llm=llm,
            similarity_top_k=10,  # Increased to retrieve more relevant chunks for better context
            streaming=True,
            verbose=True,
        )

    try:
        query_engine = create_query_engine()
    except Exception as e:
        print(f"Error creating query engine: {e}")
        raise

    return query_engine


if __name__ == "__main__":
    import time

    def cli_progress(step, total, msg):
        print(f"[{step}/{total}] {msg}")

    query_engine = get_query_engine(progress_callback=cli_progress)
    while True:
        question = input("\nAsk a question (or 'exit'): ")
        if question.lower() == "exit":
            break
        start_time = time.time()
        search_query = question
        # Use hybrid search for retrieval
        retrieved_nodes = hybrid_search(query_engine.index_struct, search_query, k=50, alpha=0.5)

        # Wrap nodes for compatibility with rest of code
        class DummyResponse:
            pass

        response = DummyResponse()
        response.source_nodes = retrieved_nodes

        # --- Remove all post-processing for static keywords such as utilisation ---
        # Only sort for perfect semantic match, not for static field-value pairs
        def is_perfect_semantic_match(node, question):
            # Optionally, you can keep a semantic match function, or just skip this
            return 0  # No special boosting

        if hasattr(response, "source_nodes"):
            response.source_nodes = sorted(
                response.source_nodes,
                key=lambda node: is_perfect_semantic_match(node, question),
                reverse=True,
            )
        answer = ""
        # Robustly handle streaming response
        if hasattr(response, "response_gen"):
            for token in response.response_gen:
                answer += token
            print("\nAnswer:\n", answer)
        elif hasattr(response, "response"):
            print("\nAnswer:\n", response.response)
        else:
            print("\nAnswer:\n", str(response))
        elapsed = time.time() - start_time
        print(f"\nTime taken: {elapsed:.2f} seconds")
        # Debug: print retrieved context
        if hasattr(response, "source_nodes"):
            print("\n[DEBUG] Retrieved context for this answer:")
            for i, node in enumerate(response.source_nodes, 1):
                print(f"--- Source Node {i} ---")
                print(f"Source file: {node.node.metadata.get('source')}")
                print(node.node.text)
                print()
        print("\n---\n")
