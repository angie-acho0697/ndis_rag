from functools import lru_cache
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List

import faiss
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml


class Llama3QASystem:
    def __init__(
        self,
        file_path: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        model_name: str = None,
        embedding_model: str = None,
    ):
        """
        QA System for NDIS data using any supported LLM.
        Loads model_name, embedding_model, chunk_size, and chunk_overlap from config.yaml if not provided.
        """
        # Load config.yaml if needed
        if any(x is None for x in [model_name, embedding_model, chunk_size, chunk_overlap]):
            try:
                with open("config.yaml", "r") as f:
                    config = yaml.safe_load(f)
                if model_name is None:
                    model_name = config.get("model_name", "mistral")
                if embedding_model is None:
                    embedding_model = config.get(
                        "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                    )
                if chunk_size is None:
                    chunk_size = config.get("chunk_size", 3000)
                if chunk_overlap is None:
                    chunk_overlap = config.get("chunk_overlap", 100)
            except Exception as e:
                print(f"Warning: Could not load config.yaml: {e}")
                if model_name is None:
                    model_name = "mistral"
                if embedding_model is None:
                    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                if chunk_size is None:
                    chunk_size = 3000
                if chunk_overlap is None:
                    chunk_overlap = 100

        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.vectorstore = None
        self.qa_chain = None
        self.model_name = model_name
        self.tables = {}  # Store tables for direct access

        t0 = time.time()
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        print(f"Embedding model loaded in {time.time() - t0:.2f} seconds.")

        self._setup_llm()

    def _setup_llm(self):
        t0 = time.time()
        try:
            print(f"Setting up LLM: {self.model_name}")
            self.llm = OllamaLLM(
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

    def _extract_tables_from_text(self, text):
        """Extract tables from text and store them for direct access."""
        tables = []
        current_table = []
        in_table = False

        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for table start patterns
            if (
                (line.startswith("+") and "-" in line)
                or (line.startswith("|") and "|" in line)
                or (line.startswith("┌") and "─" in line)
                or (line.startswith("╔") and "═" in line)
            ):
                in_table = True
                current_table = [line]

                # Look ahead for table end
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:  # Empty line marks table end
                        break
                    if (
                        (next_line.startswith("+") and "-" in next_line)
                        or (next_line.startswith("|") and "|" in next_line)
                        or (next_line.startswith("┌") and "─" in next_line)
                        or (next_line.startswith("╔") and "═" in next_line)
                    ):
                        current_table.append(next_line)
                    else:
                        current_table.append(next_line)
                    j += 1

                if current_table:
                    table_text = "\n".join(current_table)
                    # Convert table text to pandas DataFrame
                    try:
                        # Split into rows and clean
                        rows = [row.strip().split("|") for row in table_text.split("\n")]
                        rows = [[cell.strip() for cell in row] for row in rows]

                        # Get headers and data
                        headers = rows[0]
                        data = rows[2:]  # Skip header and separator

                        # Create DataFrame
                        df = pd.DataFrame(data, columns=headers)

                        # Store table with metadata
                        table_info = {"text": table_text, "dataframe": df, "file": self.file_path}
                        tables.append(table_info)
                    except Exception as e:
                        print(f"Error converting table to DataFrame: {e}")
                i = j
            else:
                i += 1

        return tables

    def _detect_headings(self, text: str) -> List[tuple]:
        """Detect headings and their content."""
        # Common heading patterns
        heading_patterns = [
            (r"^#+\s+(.+)$", "markdown"),  # Markdown headings
            (r"^[A-Z][A-Za-z\s]+:$", "title"),  # Title case headings
            (r"^\d+\.\s+[A-Z][A-Za-z\s]+$", "numbered"),  # Numbered headings
        ]

        lines = text.split("\n")
        headings = []
        current_heading = None
        current_content = []

        for line in lines:
            is_heading = False
            for pattern, style in heading_patterns:
                if re.match(pattern, line):
                    if current_heading:
                        headings.append((current_heading, "\n".join(current_content)))
                    current_heading = line
                    current_content = []
                    is_heading = True
                    break

            if not is_heading and current_heading:
                current_content.append(line)

        if current_heading:
            headings.append((current_heading, "\n".join(current_content)))

        return headings

    def load_data(self):
        """Load and process the data file with dynamic metadata extraction for both CSV and DOCX tables. Tag chunks as 'data' or 'definition'."""
        import re
        from langchain.schema import Document
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split by 'Data:' or 'Tables:' at the start of a line
            table_splits = re.split(r"(?=^(Data:|Tables:))", content, flags=re.MULTILINE)
            all_chunks = []

            # Reconstruct chunks (since split keeps the separator as its own element)
            chunks = []
            i = 0
            while i < len(table_splits):
                if table_splits[i].startswith("Data:") or table_splits[i].startswith("Tables:"):
                    chunk = table_splits[i] + (
                        table_splits[i + 1] if i + 1 < len(table_splits) else ""
                    )
                    if chunk.strip():
                        chunks.append(chunk)
                    i += 2
                else:
                    i += 1

            # Dynamically collect all possible values for each metadata field
            all_text = "\n".join(chunks)
            date_pattern = r"(\d{4}-\d{2}-\d{2}|\d{2}[A-Z]{3}\d{4})"
            state_pattern = r"\b[A-Z]{2,3}\b"
            age_pattern = r"\b\d+\s*(?:to|-|–|—)\s*\d+\b"
            all_regions = set()
            all_disabilities = set()
            for line in all_text.split("\n"):
                cells = [c.strip() for c in line.split("|")]
                if len(cells) > 4:
                    all_regions.add(cells[2])
                    all_disabilities.add(cells[3])
            all_dates = set(re.findall(date_pattern, all_text))
            all_states = set(re.findall(state_pattern, all_text))
            all_ages = set(re.findall(age_pattern, all_text))

            # For each chunk, extract metadata by matching against these sets
            for chunk in chunks:
                metadata = {}
                for date in all_dates:
                    if date in chunk:
                        metadata["date"] = date
                        break
                for state in all_states:
                    if state in chunk:
                        metadata["state"] = state
                        break
                for region in all_regions:
                    if region and region in chunk:
                        metadata["region"] = region
                        break
                for disability in all_disabilities:
                    if disability and disability in chunk:
                        metadata["disability"] = disability
                        break
                for age in all_ages:
                    if age in chunk:
                        metadata["age"] = age
                        break
                metadata["source"] = self.file_path
                # Tag as 'data' or 'definition'
                if chunk.lstrip().startswith("Data:"):
                    metadata["type"] = "data"
                elif chunk.lstrip().startswith("Tables:"):
                    metadata["type"] = "definition"
                else:
                    metadata["type"] = "unknown"
                all_chunks.append(Document(page_content=chunk, metadata=metadata))

            self.documents = all_chunks
            print(f"Total grouped chunks created: {len(all_chunks)}")
            return all_chunks
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return []

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
            # Normalize embeddings
            batch_embeddings = [emb / np.linalg.norm(emb) for emb in batch_embeddings]
            embeddings.extend(batch_embeddings)

        print(f"Embeddings generated and normalized in {time.time() - t0:.2f} seconds.")

        # Create FAISS index using langchain's implementation
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=self.embedding_model,
            metadatas=[{"source": str(i)} for i in range(len(texts))],
            normalize_L2=True,
        )

        print(f"Optimized FAISS index built in {time.time() - t0:.2f} seconds.")
        return True

    def setup_qa_chain(self):
        t0 = time.time()
        if not self.vectorstore:
            print("Vector store not initialized. Please build index first.")
            return False

        prompt_template = """
        You are a helpful AI assistant built for the National Disability Insurance Scheme (NDIS), an Australian government initiative that provides funding and support to individuals with permanent and significant disabilities.

        You answer questions based only on the provided context below. Your goal is to provide accurate, factual responses grounded in the context. If the information is not found in the context, say "The answer is not available in the provided context."

        If the context contains both data tables and definition tables:
        - Use the data table to find the answer.
        - Use the definition table to explain the meaning of any columns or terms if needed.

        If the context contains a table:
        1. Carefully read the table and identify the relevant columns and rows based on the question's criteria (such as date, disability group, age band, state, region, etc.).
        2. Filter the table rows to match all specified criteria in the question.
        3. If the question asks for a calculation (such as average, sum, or percentage), first check if the average or summary value is already present in the table (for example, in a column labeled 'average', 'Avg', 'average annualised committed support', or similar). If the value is present, return it directly without recalculating. Only perform calculations if the value is not explicitly available in the table.
        4. Clearly show your work, including which rows and columns were used, and explain your reasoning step by step.
        5. If a unique value is found, return it directly.
        6. If the answer cannot be determined from the table, explicitly say so.

        You must not fabricate information, speculate, or express opinions. When using the context, cite or paraphrase the relevant part clearly.

        If appropriate, structure the answer in bullet points or sections.

        Context:
        {context}

        Question:
        {question}

        Instructions:
        - Use only the above context to answer.
        - If the answer cannot be determined from the context, explicitly say so.
        - Do not guess or provide irrelevant information.
        - Be concise but thorough.
        - For calculations, show your work and the table rows used.

        Answer:
        """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 50,
                "lambda_mult": 0.7,
                "score_threshold": 0.5,
            },
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        print(
            f"QA system is ready to answer questions using {self.model_name}! (setup in {time.time() - t0:.2f} seconds)"
        )
        return True

    def extract_answer_from_table(self, table_text, filters):
        # Parse the table into a DataFrame
        lines = [line for line in table_text.split("\n") if "|" in line]
        if len(lines) < 2:
            return None
        headers = [h.strip() for h in lines[0].split("|") if h.strip()]
        data = [[cell.strip() for cell in row.split("|") if cell.strip()] for row in lines[2:]]
        if not data:
            return None
        try:
            df = pd.DataFrame(data, columns=headers)
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return None
        # Apply filters
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col].str.contains(val, case=False, na=False)]
        if not df.empty:
            return df.to_dict(orient="records")
        return None

    @lru_cache(maxsize=128)
    def answer_question(self, question: str) -> Dict[str, Any]:
        t0 = time.time()
        if not self.qa_chain:
            print("QA system not initialized. Run setup first.")
            return {"answer": "System not ready. Please initialize the QA system first."}
        try:
            # --- Dynamic keyword extraction using spaCy NER ---
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(question)
                keywords = set([ent.text for ent in doc.ents])
                for token in doc:
                    if token.is_title or token.like_num or "-" in token.text:
                        keywords.add(token.text)
                keywords = [
                    kw
                    for kw in keywords
                    if len(kw) > 1
                    and kw.lower()
                    not in {
                        "what",
                        "was",
                        "the",
                        "in",
                        "for",
                        "of",
                        "and",
                        "to",
                        "is",
                        "on",
                        "by",
                        "with",
                        "as",
                        "at",
                        "from",
                        "do",
                        "can",
                        "we",
                        "it",
                        "all",
                        "are",
                        "be",
                        "an",
                        "or",
                        "if",
                        "not",
                        "so",
                        "but",
                    }
                ]
                print(f"Extracted keywords from question: {keywords}")
            except Exception as e:
                print(f"spaCy not available or model not loaded: {e}")
                import re
                keywords = re.findall(r"([A-Z][a-z]+|[0-9]+(?:-[0-9]+)?)", question)
                print(f"Fallback keyword extraction: {keywords}")

            age_groups = self.extract_age_groups(question)
            keywords.extend(age_groups)
            print(f"Final keywords (with age groups): {keywords}")

            METRIC_KEYWORDS = [
                "utilised",
                "utilisation",
                "utilisation rate",
                "average annualised committed support",
                "committed support",
                "budget",
                "support",
                "provided",
                "spent",
                "total",
                "amount",
                "avganldcmtdsuppbdgt",
                "prtcpntcnt",
                "Utlstn",
            ]
            question_lower = question.lower()
            for metric in METRIC_KEYWORDS:
                if metric in question_lower:
                    keywords.append(metric)
            print(f"Final keywords (with metrics): {keywords}")

            filters = {}
            col_map = {
                "date": ["RprtDt", "Date"],
                "state": ["StateCd", "State"],
                "region": ["SrvcDstrctNm", "Region"],
                "disability": ["DsbltyGrpNm", "Disability"],
                "age": ["AgeBnd", "Age"],
                "support": ["AvgAnlsdCmtdSuppBdgt", "Support", "Utlstn", "Utilisation"],
            }
            import re
            date_match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{2}[A-Z]{3}\d{4})", question)
            if date_match:
                filters["RprtDt"] = date_match.group(1)
            age_match = re.search(r"(\d+\s*(?:-|to|–|—)\s*\d+)", question)
            if age_match:
                filters["AgeBnd"] = (
                    age_match.group(1)
                    .replace(" ", "")
                    .replace("to", "-")
                    .replace("–", "-")
                    .replace("—", "-")
                )
            state_match = re.search(r"\b([A-Z]{2,3})\b", question)
            if state_match:
                filters["StateCd"] = state_match.group(1)
            for kw in keywords:
                if "palsy" in kw.lower():
                    filters["DsbltyGrpNm"] = kw
            print(f"Filters for table lookup: {filters}")

            # --- Retrieve relevant data and definition chunks ---
            data_chunks = [doc for doc in self.documents if doc.metadata.get("type") == "data"]
            definition_chunks = [doc for doc in self.documents if doc.metadata.get("type") == "definition"]

            # Filter data chunks for those matching the filters
            relevant_data_chunks = []
            for doc in data_chunks:
                if all(str(v).lower() in doc.page_content.lower() for v in filters.values()):
                    relevant_data_chunks.append(doc)

            # Find which columns are referenced in the question or answer
            columns_needed = set()
            for metric in METRIC_KEYWORDS:
                if metric in question_lower:
                    columns_needed.add(metric)
            # Also add any columns found in the filtered data chunk headers
            for doc in relevant_data_chunks:
                header_line = doc.page_content.split("\n")[0]
                for col in header_line.split("|"):
                    if col.strip():
                        columns_needed.add(col.strip())

            # Retrieve relevant definition chunks for those columns
            relevant_definition_chunks = []
            for col in columns_needed:
                for doc in definition_chunks:
                    if col.lower() in doc.page_content.lower():
                        relevant_definition_chunks.append(doc)

            # Compose context: data first, then definitions
            context = "\n\n".join([doc.page_content for doc in relevant_data_chunks + relevant_definition_chunks])

            # Use the LLM to generate an answer based on this context and the question
            result = self.qa_chain({"query": question, "context": context})
            return {
                "answer": result.get("result", "No answer found."),
                "source_documents": relevant_data_chunks + relevant_definition_chunks,
            }
        except Exception as e:
            print(f"Error answering question: {e}")
            return {"answer": f"Error processing question: {str(e)}", "source_documents": []}

    @staticmethod
    def normalize(text):
        import re

        text = text.lower()
        text = re.sub(r"\bto\b", "-", text)
        text = re.sub(r"[–—]", "-", text)
        text = re.sub(r"[^a-z0-9 \-]", "", text)  # dash at end or escaped
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def fuzzy_in(keyword, text):
        return Llama3QASystem.normalize(keyword) in Llama3QASystem.normalize(text)

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
        return re.sub(r"^()?\\[.*?\\] *<<SYS>> *", "", answer).strip()

    def extract_age_groups(self, question):
        # Find patterns like 0-8, 0 to 8, 0–8, 0 — 8, etc.
        patterns = re.findall(r"(\d+\s*(?:-|to|–|—)\s*\d+)", question)
        return [
            p.replace(" ", "").replace("to", "-").replace("–", "-").replace("—", "-")
            for p in patterns
        ]

    def answer_question_tapas(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using TAPAS Table QA pipeline over all table chunks.
        - For each table chunk, parse as DataFrame and use TAPAS to answer.
        - If a confident answer is found, return it.
        - Otherwise, fall back to the LLM.
        Requires: transformers, pandas
        """
        try:
            from transformers import pipeline

            tapas_qa = pipeline(
                "table-question-answering", model="google/tapas-large-finetuned-wtq"
            )
        except ImportError:
            print("transformers not installed. Please install with: pip install transformers")
            return self.answer_question(question)
        except Exception as e:
            print(f"Error loading TAPAS pipeline: {e}")
            return self.answer_question(question)

        for doc in self.documents:
            # Only process chunks that look like tables
            if "|" in doc.page_content and any(
                h in doc.page_content for h in ["RprtDt", "StateCd", "DsbltyGrpNm", "AgeBnd"]
            ):
                lines = [line for line in doc.page_content.split("\n") if "|" in line]
                if len(lines) < 2:
                    continue
                headers = [h.strip() for h in lines[0].split("|") if h.strip()]
                data = [
                    [cell.strip() for cell in row.split("|") if cell.strip()] for row in lines[2:]
                ]
                if not data:
                    continue
                try:
                    import pandas as pd

                    df = pd.DataFrame(data, columns=headers)
                    result = tapas_qa(table=df, query=question)
                    # TAPAS returns a dict with 'answer' and 'aggregator'
                    answer = result.get("answer", None)
                    if answer and answer.lower() not in ["none", "nan", ""]:
                        return {"answer": f"[TAPAS] {answer}", "source_documents": [doc]}
                except Exception as e:
                    print(f"TAPAS error on table: {e}")
                    continue
        # Fallback to LLM
        return self.answer_question(question)
