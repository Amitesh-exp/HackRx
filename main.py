# import os
# import requests
# import uuid
# import tempfile
# import hashlib
# from typing import List
# import shutil

# from fastapi import FastAPI, Depends, HTTPException, status, Body, APIRouter
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel, Field, HttpUrl
# from dotenv import load_dotenv

# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Chroma

# # --- Load environment variables ---
# load_dotenv()

# # We will now rely on the library to automatically find the key from the .env file
# if not os.getenv("GOOGLE_API_KEY"):
#     raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# # --- FastAPI App and Router Setup ---
# app = FastAPI(
#     title="Retrieval System API",
#     description="API to answer questions about a document using a RAG system with MMR + Similarity hybrid search."
# )

# # Create a router with the /api/v1 prefix
# router = APIRouter(prefix="/api/v1")

# # --- Authentication ---
# security = HTTPBearer()
# # new
# EXPECTED_TOKEN = os.getenv("EXPECTED_TOKEN")
# if not EXPECTED_TOKEN:
#     raise ValueError("EXPECTED_TOKEN not found in environment variables.")
    
# async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     if not credentials or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid or missing authentication token",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     return credentials

# # --- Models ---
# class QueryRequest(BaseModel):
#     documents: HttpUrl
#     questions: List[str]

# class QueryResponse(BaseModel):
#     answers: List[str]

# # --- LLM and Embedding Setup ---
# # Removed explicit key passing; the library will find it in the environment.
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, max_output_tokens=1024)
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# RAG_PROMPT_TEMPLATE = """
# CONTEXT:
# {context}

# QUESTION:
# {question}

# Answer the question clearly and concisely using the information in the context. Do not mention the context or refer to the document. Your answer should be not more than 45-55 words. If the answer is not available, say "The document does not contain this information."
# """
# rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# # --- Vector Cache ---
# # Use a local directory for persistent caching. This is better for deployment.
# VECTOR_CACHE_DIR = "persistent_cache"
# os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)

# def get_cache_path_for_pdf(pdf_bytes: bytes) -> str:
#     return os.path.join(VECTOR_CACHE_DIR, hashlib.md5(pdf_bytes).hexdigest())

# def build_prompt(docs, question):
#     context = "\n\n".join(doc.page_content for doc in docs)
#     return rag_prompt.format(context=context, question=question)

# def create_retrievers(pdf_path: str, pdf_bytes: bytes):
#     cache_path = get_cache_path_for_pdf(pdf_bytes)
#     if os.path.exists(cache_path):
#         vector_store = Chroma(persist_directory=cache_path, embedding_function=embeddings)
#     else:
#         loader = PyPDFLoader(pdf_path)
#         documents = loader.load()
#         full_text = " ".join(doc.page_content for doc in documents)
#         # Replaced NLTKTextSplitter with a more modern and reliable splitter
#         splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
#         chunks = splitter.create_documents([full_text])
#         vector_store = Chroma.from_documents(
#             documents=chunks,
#             embedding=embeddings,
#             persist_directory=cache_path
#         )

#     retriever_sim = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
#     retriever_mmr = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

#     return retriever_sim, retriever_mmr

# # --- Endpoint (now using the router) ---
# @router.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
# async def process_document_and_answer_questions(payload: QueryRequest = Body(...)):
#     temp_pdf_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pdf")

#     try:
#         response = requests.get(str(payload.documents))
#         response.raise_for_status()
#         pdf_bytes = response.content

#         with open(temp_pdf_path, 'wb') as f:
#             f.write(pdf_bytes)

#         retriever_sim, retriever_mmr = create_retrievers(temp_pdf_path, pdf_bytes)
#         answers = []

#         for question in payload.questions:
#             docs_sim = retriever_sim.invoke(question)
#             docs_mmr = retriever_mmr.invoke(question)
            
#             all_docs = {doc.page_content: doc for doc in docs_sim + docs_mmr}.values()

#             prompt = build_prompt(list(all_docs), question)
#             answer = llm.invoke(prompt)
#             answers.append(answer.content.strip())

#         return QueryResponse(answers=answers)

#     except requests.exceptions.RequestException as e:
#         raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
#     finally:
#         if os.path.exists(temp_pdf_path):
#             os.remove(temp_pdf_path)

# # --- Health Check (now using the router) ---
# @router.get("/", tags=["Health Check"])
# def read_root():
#     return {"status": "ok", "message": "Welcome to the Retrieval System API v1!"}

# # --- Cache Management Endpoint ---
# @router.delete("/clear-cache", tags=["Cache Management"], dependencies=[Depends(verify_token)])
# def clear_cache():
#     """
#     Deletes all cached vector stores from the persistent cache directory.
#     """
#     try:
#         if os.path.exists(VECTOR_CACHE_DIR):
#             shutil.rmtree(VECTOR_CACHE_DIR)
#             os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
#             return {"status": "ok", "message": "Cache cleared successfully."}
#         return {"status": "ok", "message": "Cache directory did not exist."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")

# # Include the router in the main FastAPI app
# app.include_router(router)



import os
import requests
import uuid
import tempfile
import hashlib
from typing import List, Dict
import shutil
import re
import numpy as np

from fastapi import FastAPI, Depends, HTTPException, status, Body, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# --- Scikit-learn Imports for TF-IDF ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load environment variables ---
load_dotenv()

# We will now rely on the library to automatically find the key from the .env file
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Fetch the authentication token from environment variables for better security
EXPECTED_TOKEN = os.getenv("EXPECTED_TOKEN")
if not EXPECTED_TOKEN:
    raise ValueError("EXPECTED_TOKEN not found in environment variables. Please set it in your .env file.")

# --- FastAPI App and Router Setup ---
app = FastAPI(
    title="Retrieval System API",
    description="API to answer questions about a document using a Hybrid RAG pipeline."
)

# Create a router with the /api/v1 prefix
router = APIRouter(prefix="/api/v1")

# --- Authentication ---
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- LLM and Embedding Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, max_output_tokens=1024)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Final, Intelligent RAG Prompt ---
RAG_PROMPT_TEMPLATE = """
You are an expert insurance policy analyst. Your task is to provide a precise and definitive answer to the user's QUESTION based *only* on the provided CONTEXT.

Your reasoning process must follow these rules:
1.  **Identify the Question's Core Intent:** First, understand the specific type of information the user wants. For questions about time periods, differentiate carefully. For example, "waiting period for any illness" is different from "waiting period for pre-existing diseases".
2.  **Scan for Specifics:** Search the CONTEXT for concrete details that match the exact intent. Prioritize explicit numbers (like '30 days' or '180 days') and clear 'Exclusion' clauses over general definitions.
3.  **Handle Ambiguity with Precision:** If the CONTEXT contains multiple, seemingly relevant time periods, you must choose the one that most accurately corresponds to the user's specific question. For "waiting period for any illness," find the 'initial waiting period' clause. For "grace period," find the clause about premium payment after the due date.
4.  **Synthesize a Definitive Answer:** Based on the single best piece of information you've identified, construct a clear, detailed, and unambiguous final answer of about 30-50 words.
5.  **Default to No Coverage:** If the question asks about coverage for a specific item (like "dental") and you cannot find a clause that explicitly states it is covered, you must conclude that it is not covered.
6.  **Speak Directly:** Formulate your answer as a direct statement. Do not use phrases like "The provided text states," "According to the document," or "Based on the context."

CONTEXT:
{context}

QUESTION:
{question}

Final Answer:
"""
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
rag_chain = rag_prompt | llm | StrOutputParser()

# --- TF-IDF Helper Function ---
def process_text_for_tfidf(text: str) -> str:
    processed_text = text.lower()
    return re.sub(r'[^\w\s]', '', processed_text)

# --- Retrieval Setup ---
def setup_retrieval_components(pdf_path: str):
    """
    Loads, splits, and indexes a document for both semantic and keyword search.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    full_text = " ".join(doc.page_content for doc in documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.create_documents([full_text])

    # 1. Setup Chroma for semantic search
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever_sim = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 12})
    retriever_mmr = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 12})

    # 2. Setup TF-IDF for keyword search
    processed_chunks = [process_text_for_tfidf(chunk.page_content) for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(processed_chunks)

    return retriever_sim, retriever_mmr, vectorizer, tfidf_matrix, chunks

def perform_hybrid_retrieval(question: str, retriever_sim, retriever_mmr, vectorizer, tfidf_matrix, chunks):
    """
    Performs retrieval using all three methods and combines the results.
    """
    # 1. Semantic search
    docs_sim = retriever_sim.invoke(question)
    docs_mmr = retriever_mmr.invoke(question)

    # 2. TF-IDF keyword search
    processed_query = process_text_for_tfidf(question)
    query_embedding = vectorizer.transform([processed_query])
    similarities = cosine_similarity(tfidf_matrix, query_embedding)
    ranked_indices = np.argsort(similarities[:, 0])[::-1]
    docs_tfidf = [chunks[i] for i in ranked_indices[:5]]

    print(f"\n[QUESTION]: {question}")
    print(f"SIMILARITY DOCS: {len(docs_sim)}")
    print(f"MMR DOCS: {len(docs_mmr)}")
    print(f"TF-IDF DOCS: {len(docs_tfidf)}")

    # 3. Combine and deduplicate results
    all_docs = docs_sim + docs_mmr + docs_tfidf
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    return unique_docs

# --- Main RAG Endpoint ---
@router.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def process_document_and_answer_questions(payload: QueryRequest = Body(...)):
    temp_pdf_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pdf")
    try:
        response = requests.get(str(payload.documents))
        response.raise_for_status()

        with open(temp_pdf_path, 'wb') as f:
            f.write(response.content)

        retriever_sim, retriever_mmr, vectorizer, tfidf_matrix, chunks = setup_retrieval_components(temp_pdf_path)
        answers = []

        for question in payload.questions:
            # 1. Retrieve a wide net of documents using hybrid search
            retrieved_docs = perform_hybrid_retrieval(question, retriever_sim, retriever_mmr, vectorizer, tfidf_matrix, chunks)
            retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # 2. Generate the final answer using the single, intelligent prompt
            final_answer = rag_chain.invoke({
                "context": retrieved_context,
                "question": question
            })

            answers.append(final_answer)

        return QueryResponse(answers=answers)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

# --- Health Check ---
@router.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the Retrieval System API v1!"}

# Include the router in the main FastAPI app
app.include_router(router)
