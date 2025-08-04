import os
import requests
import uuid
import tempfile
import hashlib
from typing import List
import shutil
import re

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
    description="API to answer questions about a document using a RAG system with MMR + Similarity hybrid search."
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

# --- Models ---
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- LLM and Embedding Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, max_output_tokens=1024)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- "Mixture of Experts" Prompt Engineering ---

# Expert 1: The Direct Analyst (for simple, direct answers)
PROMPT_DIRECT = PromptTemplate.from_template("""
Based on the CONTEXT, provide a direct and concise answer to the QUESTION.

CONTEXT:
{context}

QUESTION:
{question}

Answer:
""")
chain_direct = PROMPT_DIRECT | llm | StrOutputParser()

# Expert 2: The Detail-Oriented Analyst (for finding specific numbers and durations)
PROMPT_DETAILED = PromptTemplate.from_template("""
You are a meticulous insurance policy analyst. Based *only* on the CONTEXT, provide a precise and detailed answer to the QUESTION.
Prioritize finding and including specific numbers, amounts, and durations (e.g., '30 days', '180 days').

CONTEXT:
{context}

QUESTION:
{question}

Answer:
""")
chain_detailed = PROMPT_DETAILED | llm | StrOutputParser()

# Expert 3: The Corrective Analyst (for handling term mismatches like "Restore" vs "Recovery")
PROMPT_CORRECTIVE = PromptTemplate.from_template("""
You are a smart insurance policy analyst. Your task is to answer the QUESTION based on the CONTEXT.
If the QUESTION uses a term not found in the CONTEXT, but a very similar term is present (e.g., user asks for 'Restore Benefit' but the context has 'Recovery Benefit'), you MUST point this out and answer based on the term found in the context.

CONTEXT:
{context}

QUESTION:
{question}

Answer:
""")
chain_corrective = PROMPT_CORRECTIVE | llm | StrOutputParser()

# The Final Judge: Synthesizes the best answer from the three experts
PROMPT_SYNTHESIS = PromptTemplate.from_template("""
You are the final judge. You will be given a QUESTION, the original CONTEXT, and three candidate answers from different expert analysts.
Your task is to synthesize these into a single, final, high-quality answer.

Follow these steps:
1.  Review all three candidate answers.
2.  Identify the answer that is the most accurate, complete, and relevant to the user's question. The detailed answer is often the best starting point, but the corrective answer may have crucial nuance.
3.  Use the original CONTEXT to verify all facts and details.
4.  Construct a final, polished answer that combines the best elements of all candidates.

CONTEXT:
{context}

QUESTION:
{question}

CANDIDATE ANSWER 1 (Direct):
{answer_direct}

CANDIDATE ANSWER 2 (Detailed):
{answer_detailed}

CANDIDATE ANSWER 3 (Corrective):
{answer_corrective}

FINAL BEST ANSWER:
""")
chain_synthesis = PROMPT_SYNTHESIS | llm | StrOutputParser()


def create_retrievers(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    full_text = " ".join(doc.page_content for doc in documents)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.create_documents([full_text])
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever_sim = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    retriever_mmr = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 8})

    return retriever_sim, retriever_mmr

# --- Endpoint (now using the router) ---
@router.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def process_document_and_answer_questions(payload: QueryRequest = Body(...)):
    temp_pdf_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pdf")
    try:
        response = requests.get(str(payload.documents))
        response.raise_for_status()
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(response.content)

        retriever_sim, retriever_mmr = create_retrievers(temp_pdf_path)
        answers = []

        for question in payload.questions:
            # 1. Retrieve a wide net of documents
            docs_sim = retriever_sim.invoke(question)
            docs_mmr = retriever_mmr.invoke(question)
            all_docs = {doc.page_content: doc for doc in docs_sim + docs_mmr}.values()
            retrieved_context = "\n\n".join([doc.page_content for doc in all_docs])
            
            # 2. Run all three expert chains
            answer_direct = chain_direct.invoke({"context": retrieved_context, "question": question})
            answer_detailed = chain_detailed.invoke({"context": retrieved_context, "question": question})
            answer_corrective = chain_corrective.invoke({"context": retrieved_context, "question": question})

            # 3. Use the final judge to synthesize the best answer
            final_answer = chain_synthesis.invoke({
                "context": retrieved_context,
                "question": question,
                "answer_direct": answer_direct,
                "answer_detailed": answer_detailed,
                "answer_corrective": answer_corrective
            })
            
            answers.append(final_answer)

        return QueryResponse(answers=answers)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


# --- Health Check (now using the router) ---
@router.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the Retrieval System API v1!"}


# Include the router in the main FastAPI app
app.include_router(router)





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
