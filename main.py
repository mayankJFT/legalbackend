import json
import time
import uuid
import logging
import random
from typing import Dict, List, Optional, Any, AsyncGenerator
import os
import re
import asyncio
from contextlib import asynccontextmanager
import traceback
from datetime import datetime

import tiktoken
import redis.asyncio as redis_async
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain and Vector Store Imports
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# Rate limiting
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("NyayaGPT-API")

# === API Models ===
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None

class QueryRequest(BaseModel):
    query: str
    model_name: str = "gpt-4o-mini"  # Default to faster, cheaper model
    conversation_id: Optional[str] = None
    strategy: str = "simple"  # Default to simpler strategy for speed
    max_tokens: int = 1024  # Reduced default for faster responses
    temperature: float = 0.2  # Lower temperature for more deterministic responses
    stream: bool = True  # Enable streaming by default for faster perceived response
    include_history: bool = True  # Whether to include conversation history

class ResponseMetadata(BaseModel):
    model: str
    strategy: str
    chunks_retrieved: int
    tokens_used: int
    processing_time: float
    conversation_id: str

class QueryResponse(BaseModel):
    response: str
    metadata: ResponseMetadata
    context_sources: List[Dict[str, str]] = []

class HealthResponse(BaseModel):
    status: str
    version: str
    available_models: List[str]

class BulkDeleteRequest(BaseModel):
    message_indices: List[int]

# === Configuration with Validation ===
def validate_environment():
    """Validate required environment variables"""
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    logger.info("Environment validation passed")

# Call validation on startup
try:
    validate_environment()
except ValueError as e:
    logger.error(f"Environment validation failed: {e}")
    raise

# === Redis Configuration with Fallback ===
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_TTL = int(os.getenv("REDIS_TTL", 60 * 60 * 24 * 7))  # Default 7 days
CACHE_TTL = int(os.getenv("CACHE_TTL", 60 * 60 * 24))  # Cache responses for 24 hours

# Global variables
redis_client = None
vector_store = None
redis_available = False

# === Custom Lifespan Context Manager ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize services
    global redis_client, vector_store, redis_available
    
    logger.info("Starting NyayaGPT API...")
    
    # Initialize Redis with graceful fallback
    try:
        redis_client = await init_redis()
        redis_available = True
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {str(e)}. Continuing without Redis (some features disabled)")
        redis_available = False
    
    # Initialize vector store - this is critical
    try:
        vector_store = init_vector_store()
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize vector store: {str(e)}")
        # Don't raise here to allow health check to work
        vector_store = None
    
    logger.info("NyayaGPT API startup complete")
    
    yield
    
    # Shutdown: Clean up resources
    if redis_client:
        try:
            await redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")

# === Initialize App ===
app = FastAPI(
    title="NyayaGPT API",
    description="Legal Assistant API powered by LLMs with RAG",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",  # Ensure docs are available
    redoc_url="/redoc"
)

# === Add CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Initialize Redis with Better Error Handling ===
async def init_redis():
    """Initialize Redis connection with comprehensive error handling"""
    try:
        # Parse Redis URL to check if it's valid
        if not REDIS_URL or REDIS_URL == "redis://localhost:6379":
            logger.warning("Using default Redis URL - may not be available in production")
        
        redis_instance = redis_async.from_url(
            REDIS_URL, 
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test connection with timeout
        await asyncio.wait_for(redis_instance.ping(), timeout=5.0)
        
        # Initialize rate limiter only if Redis is working
        await FastAPILimiter.init(redis_instance)
        
        return redis_instance
    except asyncio.TimeoutError:
        logger.error("Redis connection timeout")
        raise
    except Exception as e:
        logger.error(f"Redis initialization error: {str(e)}")
        raise

# === Initialize Vector Store with Better Error Handling ===
def init_vector_store():
    """Initialize Pinecone vector store with comprehensive error handling"""
    try:
        # Get and validate API key
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = os.getenv("PINECONE_INDEX_NAME", "2025-judgements-index")
        
        logger.info(f"Attempting to connect to Pinecone index: {index_name}")
        
        # Check if index exists
        try:
            index_info = pc.describe_index(index_name)
            logger.info(f"Found Pinecone index: {index_name}")
        except Exception as e:
            logger.error(f"Pinecone index '{index_name}' not found or inaccessible: {e}")
            raise ValueError(f"Pinecone index '{index_name}' not found or inaccessible")
        
        # Get the index
        index = pc.Index(index_name)
        
        # Test the index with a simple query
        try:
            # Test query to verify index is working
            test_stats = index.describe_index_stats()
            logger.info(f"Pinecone index stats: {test_stats}")
        except Exception as e:
            logger.error(f"Failed to query Pinecone index: {e}")
            raise
        
        # Initialize embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )
        
        # Create vector store
        vector_store = PineconeVectorStore(
            index=index, 
            embedding=embeddings
        )
        
        logger.info("Vector store initialized successfully")
        return vector_store
        
    except Exception as e:
        logger.error(f"Vector store initialization error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# === LLM Configuration with Error Handling ===
def create_llm(model_name: str, streaming: bool = False):
    """Create LLM instance with proper error handling"""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        return ChatOpenAI(
            model=model_name,
            temperature=0.1,
            max_tokens=1024,
            streaming=streaming,
            request_timeout=30,
            openai_api_key=openai_api_key
        )
    except Exception as e:
        logger.error(f"Failed to create LLM {model_name}: {e}")
        raise

AVAILABLE_MODELS = {
    "gpt-4o": lambda streaming=False: create_llm("gpt-4o", streaming),
    "gpt-4o-mini": lambda streaming=False: create_llm("gpt-4o-mini", streaming),
    "gpt-3.5-turbo": lambda streaming=False: create_llm("gpt-3.5-turbo", streaming)
}

# === Prompt Templates ===
final_prompt = PromptTemplate(
    template="""
You are NyayaGPT, a highly knowledgeable legal assistant trained on Supreme Court and High Court judgments, Indian statutes, procedural codes, and legal drafting conventions. You can answer any query related to cases, judgments, statutes, or legal principles.

Instructions:

1. If the user's input is a simple greeting or non-legal chit-chat (e.g., "hello", "hi", "how are you"), reply conversationally without legal analysis.
2. Otherwise, treat the input as a legal query. For any case, judgment, statute, or legal principle query:
   - Analyze the fact pattern or legal issue thoroughly.
   - Identify and explain all relevant statutory provisions, legal principles, and landmark judgments.
   - Structure your response issue-wise, using clear headings for each issue or legal point.
   - Provide proper citations: include case names, dates, courts, judges, section numbers, and paragraph or page references where applicable.
   - If the user requests or implies a legal document (petition, application, draft, etc.), deliver a complete, formatted template.
   - Supply a concise list of all cited cases with their titles, jurisdictions, and decision dates.

Use the following context where relevant:

Previous Conversation:
{history}

Context Information (if provided; use it to enrich your analysis):
{context}

Current Question:
{question}
""",
    input_variables=["history", "context", "question"]
)

fusion_prompt = ChatPromptTemplate.from_template("""
You are an assistant skilled in legal language modeling.
Given the following user query, generate 3 different rephrasings of it as formal Indian legal questions.
Do not invent extra facts or foreign law. Just reword using Indian legal terminology.

User Query: {question}

Three Rephrasings:""")

# === Utility Functions ===
def is_simple_greeting(text):
    """Detect if input is a simple greeting that doesn't need RAG"""
    text = text.lower().strip()
    greeting_patterns = [
        r'^(hi|hello|hey|greetings|namaste|howdy)[\s\W]*$',
        r'^(good\s*(morning|afternoon|evening|day))[\s\W]*$',
        r'^(how\s*(are\s*you|is\s*it\s*going|are\s*things))[\s\W]*$',
        r'^(what\'*s\s*up)[\s\W]*$'
    ]
    
    for pattern in greeting_patterns:
        if re.match(pattern, text):
            return True
    return False

def get_greeting_response(greeting_text):
    """Generate appropriate response for simple greetings without using LLM"""
    greeting_text = greeting_text.lower().strip()
    
    if re.match(r'^(hi|hello|hey|howdy)[\s\W]*$', greeting_text):
        responses = [
            "Hello! How can I help you with legal information today?",
            "Hi there! I'm NyayaGPT, your legal assistant. What legal questions can I help you with?",
            "Hello! I'm ready to assist with your legal queries."
        ]
        return random.choice(responses)
    
    elif re.match(r'^(good\s*morning)[\s\W]*$', greeting_text):
        return "Good morning! How can I assist you with legal matters today?"
    
    elif re.match(r'^(good\s*afternoon)[\s\W]*$', greeting_text):
        return "Good afternoon! What legal questions can I help you with today?"
    
    elif re.match(r'^(good\s*evening)[\s\W]*$', greeting_text):
        return "Good evening! I'm here to help with any legal queries you might have."
    
    elif re.match(r'^(how\s*are\s*you)[\s\W]*$', greeting_text):
        return "I'm functioning well, thank you for asking! I'm ready to assist with your legal questions."
    
    elif re.match(r'^(what\'*s\s*up)[\s\W]*$', greeting_text):
        return "I'm here and ready to help with your legal queries! What can I assist you with today?"
    
    return "Hello! I'm NyayaGPT, your legal assistant. How can I help you today?"

def format_docs(docs, max_length=600):
    """Format documents with efficient length limit"""
    result = []
    for doc in docs:
        title = doc.metadata.get("title", "Untitled Document")
        url = doc.metadata.get("url", "No URL")
        result.append(f"### {title}\n**Source:** {url}\n\n{doc.page_content.strip()[:max_length]}...")
    return "\n\n".join(result)

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text with error handling"""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {str(e)}. Using approximate count.")
        return len(text) // 4

def format_conversation_history(messages, max_tokens=1000):
    """Format conversation history for inclusion in the prompt"""
    formatted_history = []
    for msg in messages:
        role = msg.get("role", "user" if "query" in msg else "assistant")
        content = msg.get("content", msg.get("query", msg.get("response", "")))
        formatted_history.append(f"{role.capitalize()}: {content}")
    
    history_text = "\n\n".join(formatted_history)
    
    try:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = enc.encode(history_text)
        if len(tokens) > max_tokens:
            tokens = tokens[-max_tokens:]
            history_text = enc.decode(tokens)
            history_text = "...\n" + history_text
    except Exception as e:
        logger.warning(f"Error processing conversation history: {e}")
        if len(history_text) > max_tokens * 4:  # Approximate token limit
            history_text = history_text[-(max_tokens * 4):]
    
    return history_text

# === Retrieval Strategies ===
def fusion_strategy(query, llm):
    """Improved fusion strategy with error handling"""
    try:
        if not vector_store:
            logger.error("Vector store not available for fusion strategy")
            return []
        
        fusion_chain = fusion_prompt | llm
        response = fusion_chain.invoke({"question": query})
        variants = [line.strip("- ") for line in response.content.strip().split("\n") if line.strip()][:2]
        variants.insert(0, query)
        
        seen = set()
        all_docs = []
        
        for variant in variants:
            try:
                for doc in vector_store.similarity_search(variant, k=3):
                    hash_ = doc.page_content[:100]
                    if hash_ not in seen:
                        seen.add(hash_)
                        all_docs.append(doc)
            except Exception as e:
                logger.warning(f"Error searching for variant '{variant}': {e}")
                continue
        
        return all_docs[:5]
    except Exception as e:
        logger.error(f"Error in fusion strategy: {e}")
        return simple_strategy(query, llm)

def simple_strategy(query, llm):
    """Direct retrieval with error handling"""
    try:
        if not vector_store:
            logger.error("Vector store not available for simple strategy")
            return []
        return vector_store.similarity_search(query, k=5)
    except Exception as e:
        logger.error(f"Error in simple strategy: {e}")
        return []

# === Redis Conversation Storage with Fallbacks ===
async def get_conversation(conversation_id):
    """Get conversation history from Redis with error handling"""
    if not redis_available or not redis_client:
        logger.debug("Redis not available - returning empty conversation history")
        return []
    
    try:
        conversation_data = await redis_client.get(f"conv:{conversation_id}")
        if conversation_data:
            return json.loads(conversation_data)
        return []
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        return []

async def save_message_to_conversation(conversation_id, message):
    """Save a single message to the conversation history with error handling"""
    if not redis_available or not redis_client:
        logger.debug("Redis not available - skipping message save")
        return
    
    try:
        conversation = await get_conversation(conversation_id)
        
        if "timestamp" not in message:
            message["timestamp"] = time.time()
        
        conversation.append(message)
        
        await redis_client.setex(
            f"conv:{conversation_id}", 
            REDIS_TTL, 
            json.dumps(conversation)
        )
    except Exception as e:
        logger.error(f"Error saving message to conversation: {str(e)}")

# === Cache Helper Functions ===
async def get_cached_response(query: str, model_name: str, strategy: str):
    """Get cached response if available"""
    if not redis_available or not redis_client:
        return None
    
    try:
        cache_key = f"cache:{hash(f'{query}:{model_name}:{strategy}')}"
        cached = await redis_client.get(cache_key)
        
        if cached:
            logger.info(f"Cache hit for query: {query[:30]}...")
            return json.loads(cached)
        return None
    except Exception as e:
        logger.error(f"Error retrieving from cache: {str(e)}")
        return None

async def cache_response(query: str, model_name: str, strategy: str, response_data: dict):
    """Cache response for future use"""
    if not redis_available or not redis_client:
        return
    
    try:
        cache_key = f"cache:{hash(f'{query}:{model_name}:{strategy}')}"
        await redis_client.setex(
            cache_key,
            CACHE_TTL,
            json.dumps(response_data)
        )
        logger.info(f"Cached response for query: {query[:30]}...")
    except Exception as e:
        logger.error(f"Error caching response: {str(e)}")

# === Core Query Processing ===
async def process_query(query_request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a query with improved error handling"""
    start_time = time.time()
    conversation_id = query_request.conversation_id or str(uuid.uuid4())
    
    try:
        # Check if vector store is available
        if not vector_store:
            raise HTTPException(
                status_code=503,
                detail="Vector store not available. Please contact support."
            )
        
        # Save user query to conversation
        user_message = {
            "role": "user",
            "content": query_request.query,
            "timestamp": time.time()
        }
        await save_message_to_conversation(conversation_id, user_message)
        
        # Check cache for non-streaming requests
        if not query_request.stream:
            cached = await get_cached_response(
                query_request.query, 
                query_request.model_name,
                query_request.strategy
            )
            if cached:
                cached["metadata"]["conversation_id"] = conversation_id
                
                assistant_message = {
                    "role": "assistant",
                    "content": cached["response"],
                    "timestamp": time.time()
                }
                await save_message_to_conversation(conversation_id, assistant_message)
                
                return QueryResponse(**cached)
        
        # Validate model
        if query_request.model_name not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Model {query_request.model_name} not available. Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        
        # Initialize LLM
        try:
            llm = AVAILABLE_MODELS[query_request.model_name](streaming=query_request.stream)
            llm.temperature = query_request.temperature
            llm.max_tokens = query_request.max_tokens
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize language model")
        
        # Get conversation history
        conversation_history = ""
        if query_request.include_history:
            past_messages = await get_conversation(conversation_id)
            if len(past_messages) > 1:
                conversation_history = format_conversation_history(past_messages[:-1])
        
        # Fast-path for simple greetings
        if is_simple_greeting(query_request.query):
            greeting_response = get_greeting_response(query_request.query)
            
            assistant_message = {
                "role": "assistant",
                "content": greeting_response,
                "timestamp": time.time()
            }
            await save_message_to_conversation(conversation_id, assistant_message)
            
            duration = time.time() - start_time
            
            response = QueryResponse(
                response=greeting_response,
                metadata=ResponseMetadata(
                    model="fast-path-greeting",
                    strategy="direct",
                    chunks_retrieved=0,
                    tokens_used=0,
                    processing_time=round(duration, 2),
                    conversation_id=conversation_id
                ),
                context_sources=[]
            )
            
            return response
        
        # Retrieve documents
        retrieve_fn = fusion_strategy if query_request.strategy == "fusion" else simple_strategy
        
        try:
            docs = retrieve_fn(query_request.query, llm)
        except Exception as e:
            logger.warning(f"Error in retrieval: {str(e)}. Falling back to simple strategy.")
            docs = simple_strategy(query_request.query, llm)
        
        # Format context
        context = format_docs(docs, max_length=400)
        
        # Create prompt
        prompt = final_prompt.format(
            history=conversation_history,
            context=context, 
            question=query_request.query
        )
        
        # Count tokens
        tokens_used = count_tokens(prompt, query_request.model_name)
        
        # Generate response
        try:
            parser = StrOutputParser()
            answer = (llm | parser).invoke(prompt)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail="Error generating response")
        
        # Save assistant response
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "timestamp": time.time()
        }
        await save_message_to_conversation(conversation_id, assistant_message)
        
        # Create sources list
        sources = [
            {
                "title": doc.metadata.get("title", "Untitled"),
                "url": doc.metadata.get("url", "No URL"),
                "snippet": doc.page_content[:100] + "..."
            }
            for doc in docs
        ]
        
        duration = time.time() - start_time
        
        response = QueryResponse(
            response=answer,
            metadata=ResponseMetadata(
                model=query_request.model_name,
                strategy=query_request.strategy,
                chunks_retrieved=len(docs),
                tokens_used=tokens_used,
                processing_time=round(duration, 2),
                conversation_id=conversation_id
            ),
            context_sources=sources
        )
        
        # Cache response if not streaming
        if not query_request.stream:
            await cache_response(
                query_request.query,
                query_request.model_name,
                query_request.strategy,
                response.dict()
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# === Streaming Response Generator ===
async def generate_streaming_response(query_request: QueryRequest, background_tasks: BackgroundTasks) -> AsyncGenerator[str, None]:
    """Generate a streaming response for the query"""
    start_time = time.time()
    conversation_id = query_request.conversation_id or str(uuid.uuid4())
    
    try:
        # Check if vector store is available
        if not vector_store:
            error_msg = json.dumps({
                "error": "Vector store not available. Please contact support."
            })
            yield f"data: {error_msg}\n\n"
            return
        
        # Validate model
        if query_request.model_name not in AVAILABLE_MODELS:
            error_msg = json.dumps({
                "error": f"Model {query_request.model_name} not available. Available models: {list(AVAILABLE_MODELS.keys())}"
            })
            yield f"data: {error_msg}\n\n"
            return
        
        # Save user query
        user_message = {
            "role": "user",
            "content": query_request.query,
            "timestamp": time.time()
        }
        await save_message_to_conversation(conversation_id, user_message)
        
        # Initialize LLM
        try:
            llm = AVAILABLE_MODELS[query_request.model_name](streaming=True)
            llm.temperature = query_request.temperature
            llm.max_tokens = query_request.max_tokens
        except Exception as e:
            error_msg = json.dumps({"error": f"Failed to initialize model: {str(e)}"})
            yield f"data: {error_msg}\n\n"
            return
        
        # Get conversation history
        conversation_history = ""
        if query_request.include_history:
            past_messages = await get_conversation(conversation_id)
            if len(past_messages) > 1:
                conversation_history = format_conversation_history(past_messages[:-1])
        
        # Fast-path for greetings
        if is_simple_greeting(query_request.query):
            greeting_response = get_greeting_response(query_request.query)
            
            yield f"data: {json.dumps({'chunk': greeting_response, 'full': greeting_response})}\n\n"
            
            assistant_message = {
                "role": "assistant",
                "content": greeting_response,
                "timestamp": time.time()
            }
            await save_message_to_conversation(conversation_id, assistant_message)
            
            duration = time.time() - start_time
            completion_data = {
                "done": True,
                "metadata": {
                    "model": "fast-path-greeting",
                    "strategy": "direct",
                    "chunks_retrieved": 0,
                    "tokens_used": 0,
                    "processing_time": round(duration, 2),
                    "conversation_id": conversation_id
                },
                "context_sources": []
            }
            
            yield f"data: {json.dumps(completion_data)}\n\n"
            return
        
        # Retrieve documents
        retrieve_fn = fusion_strategy if query_request.strategy == "fusion" else simple_strategy
        
        try:
            docs = retrieve_fn(query_request.query, llm)
        except Exception as e:
            logger.warning(f"Error in retrieval: {str(e)}. Falling back to simple strategy.")
            docs = simple_strategy(query_request.query, llm)
        
        # Format context
        context = format_docs(docs, max_length=400)
        
        # Create prompt
        prompt = final_prompt.format(
            history=conversation_history,
            context=context, 
            question=query_request.query
        )
        
        tokens_used = count_tokens(prompt, query_request.model_name)
        
        # Stream response
        chain = llm | StrOutputParser()
        
        full_response = ""
        try:
            async for chunk in chain.astream(prompt):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk, 'full': full_response})}\n\n"
        except Exception as e:
            error_msg = json.dumps({"error": f"Streaming error: {str(e)}"})
            yield f"data: {error_msg}\n\n"
            return
        
        # Save assistant response
        assistant_message = {
            "role": "assistant",
            "content": full_response,
            "timestamp": time.time()
        }
        await save_message_to_conversation(conversation_id, assistant_message)
        
        duration = time.time() - start_time
        
        # Create sources list
        sources = [
            {
                "title": doc.metadata.get("title", "Untitled"),
                "url": doc.metadata.get("url", "No URL"),
                "snippet": doc.page_content[:100] + "..."
            }
            for doc in docs
        ]
        
        # Send completion metadata
        completion_data = {
            "done": True,
            "metadata": {
                "model": query_request.model_name,
                "strategy": query_request.strategy,
                "chunks_retrieved": len(docs),
                "tokens_used": tokens_used,
                "processing_time": round(duration, 2),
                "conversation_id": conversation_id
            },
            "context_sources": sources
        }
        
        yield f"data: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        error_data = {
            "error": str(e),
            "full": f"I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists."
        }
        yield f"data: {error_data}\n\n"
        
        completion_data = {
            "done": True,
            "metadata": {
                "model": query_request.model_name,
                "strategy": query_request.strategy,
                "chunks_retrieved": 0,
                "tokens_used": 0,
                "processing_time": round(time.time() - start_time, 2),
                "conversation_id": conversation_id
            },
            "context_sources": [],
            "error": str(e)
        }
        
        yield f"data: {json.dumps(completion_data)}\n\n"

# === Rate Limiter Dependency ===
def get_rate_limiter():
    """Get rate limiter if Redis is available, otherwise return a no-op"""
    if redis_available:
        return RateLimiter(times=20, seconds=60)
    else:
        # Return a no-op dependency when Redis is not available
        async def no_op_limiter():
            pass
        return no_op_limiter

# === API Endpoints ===
@app.get("/", response_model=dict)
async def root():
    """Root endpoint for basic API info"""
    return {
        "message": "NyayaGPT API is running",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "query": "/query"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with service status"""
    services_status = {
        "redis": redis_available,
        "vector_store": vector_store is not None,
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "pinecone": bool(os.getenv("PINECONE_API_KEY"))
    }
    
    # Determine overall status
    critical_services = ["vector_store", "openai", "pinecone"]
    status = "healthy" if all(services_status[service] for service in critical_services) else "degraded"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        available_models=list(AVAILABLE_MODELS.keys())
    )

@app.get("/status")
async def detailed_status():
    """Detailed status endpoint for debugging"""
    return {
        "services": {
            "redis": {
                "available": redis_available,
                "url": REDIS_URL if redis_available else "Not configured"
            },
            "vector_store": {
                "available": vector_store is not None,
                "index_name": os.getenv("PINECONE_INDEX_NAME", "2025-judgements-index")
            },
            "openai": {
                "configured": bool(os.getenv("OPENAI_API_KEY")),
                "models": list(AVAILABLE_MODELS.keys())
            }
        },
        "environment": {
            "redis_ttl": REDIS_TTL,
            "cache_ttl": CACHE_TTL
        }
    }

@app.get("/clear-cache")
async def clear_cache():
    """Clear the response cache"""
    if not redis_available or not redis_client:
        raise HTTPException(
            status_code=503,
            detail="Redis not available - cache clearing not supported"
        )
    
    try:
        cursor = 0
        deleted_count = 0
        
        while True:
            cursor, keys = await redis_client.scan(cursor, match="cache:*")
            if keys:
                deleted = await redis_client.delete(*keys)
                deleted_count += deleted
            
            if cursor == 0:
                break
        
        return {
            "status": "success",
            "message": f"Cache cleared: {deleted_count} entries removed"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )

# === Generate First-Time Conversation ID ===
async def get_or_create_conversation(request: Request) -> str:
    """Get existing conversation ID from cookie or create a new one"""
    conversation_id = request.cookies.get("conversation_id")
    
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        logger.info(f"Created new conversation: {conversation_id}")
    
    return conversation_id

@app.post("/query")
async def query_endpoint(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    _: None = Depends(get_rate_limiter())  # Apply rate limiting if available
):
    """Process a legal query using the specified LLM and retrieval strategy"""
    # Auto-generate conversation ID if not provided
    if not query_request.conversation_id:
        query_request.conversation_id = await get_or_create_conversation(request)
    
    # If streaming is requested, return a streaming response
    if query_request.stream:
        response = StreamingResponse(
            generate_streaming_response(query_request, background_tasks),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
        response.set_cookie(
            key="conversation_id",
            value=query_request.conversation_id,
            httponly=True,
            max_age=30*24*60*60
        )
        
        return response
    
    # Regular non-streaming response
    try:
        response_data = await process_query(query_request, background_tasks)
        
        response = JSONResponse(content=response_data.dict())
        response.set_cookie(
            key="conversation_id",
            value=query_request.conversation_id,
            httponly=True,
            max_age=30*24*60*60
        )
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Retrieve conversation history by ID"""
    if not redis_available:
        raise HTTPException(
            status_code=503,
            detail="Conversation history not available - Redis not configured"
        )
    
    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    return {"conversation_id": conversation_id, "messages": conversation}

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation by ID"""
    if not redis_available or not redis_client:
        raise HTTPException(
            status_code=503,
            detail="Conversation management not available - Redis not configured"
        )
    
    try:
        deleted = await redis_client.delete(f"conv:{conversation_id}")
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting conversation: {str(e)}"
        )

@app.delete("/conversation/{conversation_id}/message/{message_index}")
async def delete_message(conversation_id: str, message_index: int):
    """Delete a specific message from a conversation by its index"""
    if not redis_available or not redis_client:
        raise HTTPException(
            status_code=503,
            detail="Conversation management not available - Redis not configured"
        )
    
    try:
        conversation = await get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        if message_index < 0 or message_index >= len(conversation):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid message index {message_index}"
            )
        
        removed_message = conversation.pop(message_index)
        
        await redis_client.setex(
            f"conv:{conversation_id}", 
            REDIS_TTL, 
            json.dumps(conversation)
        )
        
        return {
            "status": "success", 
            "message": f"Message at index {message_index} deleted",
            "removed": removed_message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting message: {str(e)}"
        )

@app.delete("/conversation/{conversation_id}/messages")
async def delete_multiple_messages(conversation_id: str, request: BulkDeleteRequest):
    """Delete multiple messages from a conversation by their indices"""
    if not redis_available or not redis_client:
        raise HTTPException(
            status_code=503,
            detail="Conversation management not available - Redis not configured"
        )
    
    try:
        conversation = await get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        indices = sorted(request.message_indices, reverse=True)
        
        if any(idx < 0 or idx >= len(conversation) for idx in indices):
            raise HTTPException(
                status_code=400,
                detail="One or more invalid message indices"
            )
        
        removed = []
        for idx in indices:
            removed.append(conversation.pop(idx))
        
        await redis_client.setex(
            f"conv:{conversation_id}", 
            REDIS_TTL, 
            json.dumps(conversation)
        )
        
        return {
            "status": "success", 
            "message": f"Deleted {len(indices)} messages",
            "removed": removed
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting messages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting messages: {str(e)}"
        )

# === Exception Handlers ===
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error occurred",
            "error": str(exc) if os.getenv("DEBUG") else "Internal server error"
        }
    )

# === Run Server ===
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",  # Assuming this file is named main.py
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        access_log=True,
        log_level="info"
    )
