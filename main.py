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
from langchain_core.runnables import Runnable
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document, BaseRetriever
from langchain.retrievers import EnsembleRetriever

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
    description="API to answer questions about a document using an advanced RAG pipeline."
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

# --- Custom TF-IDF Retriever Class ---
class TFIDFRetriever(BaseRetriever):
    vectorizer: TfidfVectorizer
    tfidf_matrix: np.ndarray
    chunks: List[Document]
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        processed_query = process_text_for_tfidf(query)
        query_embedding = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(self.tfidf_matrix, query_embedding)
        ranked_indices = np.argsort(similarities[:, 0])[::-1]
        return [self.chunks[i] for i in ranked_indices[:self.k]]

def process_text_for_tfidf(text: str) -> str:
    processed_text = text.lower()
    return re.sub(r'[^\w\s]', '', processed_text)

# --- Retrieval Setup ---
def setup_retrieval_pipeline(pdf_path: str) -> Runnable:
    """
    Loads, splits, and creates a full retrieval pipeline with an Ensemble retriever.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    full_text = " ".join(doc.page_content for doc in documents)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.create_documents([full_text])
    
    # 1. Setup Chroma for semantic search
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    chroma_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    # 2. Setup TF-IDF for keyword search
    processed_chunks = [process_text_for_tfidf(chunk.page_content) for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    # Convert the sparse matrix to a dense numpy array to fix the validation error
    tfidf_matrix = vectorizer.fit_transform(processed_chunks).toarray()
    tfidf_retriever = TFIDFRetriever(
        vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, chunks=chunks, k=5
    )

    # 3. Setup Ensemble Retriever to combine semantic and keyword search
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, tfidf_retriever], weights=[0.5, 0.5]
    )

    return ensemble_retriever

# --- Main RAG Endpoint ---
@router.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def process_document_and_answer_questions(payload: QueryRequest = Body(...)):
    temp_pdf_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pdf")
    try:
        response = requests.get(str(payload.documents))
        response.raise_for_status()
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(response.content)

        retrieval_pipeline = setup_retrieval_pipeline(temp_pdf_path)
        answers = []

        for question in payload.questions:
            # 1. Retrieve documents using the full pipeline
            retrieved_docs = retrieval_pipeline.invoke(question)
            retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # 2. Generate the final answer
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
