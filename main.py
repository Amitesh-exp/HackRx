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
os.environ["UVLOOP_NO"] = "1"
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
EXPECTED_TOKEN = os.getenv("EXPECTED_TOKEN")
if not EXPECTED_TOKEN:
    raise ValueError("EXPECTED_TOKEN not found in environment variables.")

app = FastAPI(title="Retrieval System API", description="Hybrid RAG API")
router = APIRouter(prefix="/api/v1")

security = HTTPBearer()
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class EvaluationRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]
    ground_truths: List[str]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, max_output_tokens=1024)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

RAG_PROMPT_TEMPLATE = """
You are an expert insurance policy analyst. Your task is to provide a precise and definitive answer to the user's QUESTION based *only* on the provided CONTEXT.
... [unchanged prompt for brevity] ...
CONTEXT:
{context}

QUESTION:
{question}

Final Answer:
"""
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
rag_chain = rag_prompt | llm | StrOutputParser()

def process_text_for_tfidf(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower())

SYNONYM_MAP = {
    "restore benefit": ["restoration of sum insured", "automatic reinstatement", "reinstatement benefit", "sum insured restoration", "recovery benefit", "recovery"],
    "dental treatment": ["oral care", "tooth extraction", "dental procedure"],
    "pre-existing diseases": ["ped", "chronic conditions", "existing health issues"],
}

def expand_query_with_synonyms(query: str) -> str:
    for key, synonyms in SYNONYM_MAP.items():
        if key in query.lower():
            query += " " + " ".join(synonyms)
    return query

def regex_snippet_search(query: str, chunks: List[Document], keywords: List[str]) -> List[Document]:
    results = []
    for chunk in chunks:
        text = chunk.page_content.lower()
        if any(re.search(rf"\\b{kw.lower()}\\b", text) for kw in keywords):
            results.append(chunk)
    return results

def setup_retrieval_components(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    full_text = " ".join(doc.page_content for doc in documents)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    chunks = splitter.create_documents([full_text])

    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever_sim = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 12})
    retriever_mmr = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 12})

    processed_chunks = [process_text_for_tfidf(chunk.page_content) for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(processed_chunks)

    return retriever_sim, retriever_mmr, vectorizer, tfidf_matrix, chunks

def perform_hybrid_retrieval(question: str, retriever_sim, retriever_mmr, vectorizer, tfidf_matrix, chunks):
    docs_sim = retriever_sim.invoke(question)
    docs_mmr = retriever_mmr.invoke(question)

    expanded_query = expand_query_with_synonyms(question)
    processed_query = process_text_for_tfidf(expanded_query)
    query_embedding = vectorizer.transform([processed_query])
    similarities = cosine_similarity(tfidf_matrix, query_embedding)
    ranked_indices = np.argsort(similarities[:, 0])[::-1]
    docs_tfidf = [chunks[i] for i in ranked_indices[:5]]

    regex_keywords = SYNONYM_MAP.get(question.lower(), []) + [question]
    docs_regex = regex_snippet_search(question, chunks, regex_keywords)

    all_docs = docs_sim + docs_mmr + docs_tfidf + docs_regex
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    return unique_docs

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
            retrieved_docs = perform_hybrid_retrieval(question, retriever_sim, retriever_mmr, vectorizer, tfidf_matrix, chunks)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            final_answer = rag_chain.invoke({"context": context, "question": question})
            answers.append(final_answer)

        return QueryResponse(answers=answers)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

@router.post("/evaluate", dependencies=[Depends(verify_token)])
async def evaluate_rag_pipeline(payload: EvaluationRequest = Body(...)):
    if len(payload.questions) != len(payload.ground_truths):
        raise HTTPException(status_code=400, detail="Mismatched questions and ground truths")

    temp_pdf_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pdf")
    try:
        response = requests.get(str(payload.documents))
        response.raise_for_status()
        with open(temp_pdf_path, 'wb') as f:
            f.write(response.content)

        retriever_sim, retriever_mmr, vectorizer, tfidf_matrix, chunks = setup_retrieval_components(temp_pdf_path)
        eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        for i, question in enumerate(payload.questions):
            docs = perform_hybrid_retrieval(question, retriever_sim, retriever_mmr, vectorizer, tfidf_matrix, chunks)
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = rag_chain.invoke({"context": context, "question": question})
            eval_data["question"].append(question)
            eval_data["answer"].append(answer)
            eval_data["contexts"].append([doc.page_content for doc in docs])
            eval_data["ground_truth"].append(payload.ground_truths[i])

        dataset = Dataset.from_dict(eval_data)
        result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_recall, context_precision])
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation error: {e}")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

@router.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the Retrieval System API v1!"}

app.include_router(router)