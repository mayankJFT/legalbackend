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
    model_name: str = "gpt-4o"  # Default to faster model
    conversation_id: Optional[str] = None
    strategy: str = "simple"  # Default to simpler strategy for speed
    max_tokens: int = 2048  # Default max tokens for response
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

# === Redis Configuration ===
REDIS_URL = os.getenv("REDIS_URL", "redis://red-d10mfebipnbc738vva60:6379")
REDIS_TTL = int(os.getenv("REDIS_TTL", 60 * 60 * 24 * 7))  # Default 7 days
CACHE_TTL = int(os.getenv("CACHE_TTL", 60 * 60 * 24))  # Cache responses for 24 hours

# Global variables
redis_client = None
vector_store = None

# === Custom Lifespan Context Manager ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize services
    global redis_client, vector_store
    
    try:
        # Initialize Redis
        redis_client = await init_redis()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        # Continue without Redis - features requiring Redis will be disabled
    
    try:
        # Initialize vector store
        vector_store = init_vector_store()
        logger.info("Vector store initialized")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise  # This is critical, so we should fail startup
    
    yield
    
    # Shutdown: Clean up resources
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")

# === Initialize App ===
app = FastAPI(
    title="NyayaGPT API",
    description="Legal Assistant API powered by LLMs with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# === Add CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Initialize Redis ===
async def init_redis():
    """Initialize Redis connection with error handling"""
    try:
        redis_instance = redis_async.from_url(REDIS_URL, decode_responses=True)
        # Test connection
        await redis_instance.ping()
        await FastAPILimiter.init(redis_instance)
        return redis_instance
    except Exception as e:
        logger.error(f"Redis initialization error: {str(e)}")
        raise

# === Initialize Vector Store ===
def init_vector_store():
    """Initialize Pinecone vector store with error handling"""
    try:
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            logger.error("Pinecone API key not found")
            raise ValueError("Pinecone API key is required")
        
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "2025-judgements-index"  
        
        if not pc.has_index(index_name):
            logger.error(f"Pinecone index '{index_name}' does not exist")
            raise ValueError(f"Pinecone index '{index_name}' does not exist")
        
        index = pc.Index(index_name)
        
        return PineconeVectorStore(
            index=index, 
            embedding=OpenAIEmbeddings(model="text-embedding-ada-002")
        )
    except Exception as e:
        logger.error(f"Vector store initialization error: {str(e)}")
        raise

# === LLM Configuration ===
AVAILABLE_MODELS = {
    "gpt-4o": lambda streaming=False: ChatOpenAI(
        model="gpt-4o", 
        temperature=0.1, 
        max_tokens=2048,
        streaming=streaming,
        request_timeout=20  # 20 seconds timeout
    ),
    "gpt-4o-mini": lambda streaming=False: ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1, 
        max_tokens=2048,
        streaming=streaming,
        request_timeout=15  # 15 seconds timeout 
    ),
    "gpt-3.5-turbo": lambda streaming=False: ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.1, 
        max_tokens=2048,
        streaming=streaming,
        request_timeout=10  # 10 seconds timeout
    )
}

# === Prompt Templates ===
# Updated the prompt template to avoid hallucination and better handle simple queries
final_prompt = PromptTemplate(
    template="""
You are NyayaGPT, a highly knowledgeable legal assistant trained on Supreme Court and High Court judgments, Indian statutes, procedural codes, and legal drafting conventions. You can answer any query related to cases, judgments, statutes, or legal principles.

Instructions:

1. If the user’s input is a simple greeting or non-legal chit-chat (e.g., “hello”, “hi”, “how are you”), reply conversationally without legal analysis.
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
    
    # Simple mapping of greeting patterns to responses
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
    
    # Default response if no specific pattern matches
    return "Hello! I'm NyayaGPT, your legal assistant. How can I help you today?"

def format_docs(docs, max_length=800):
    """Format documents with a more efficient length limit to avoid token wastage"""
    result = []
    for doc in docs:
        title = doc.metadata.get("title", "Untitled Document")
        url = doc.metadata.get("url", "No URL")
        # Use shorter excerpts to save tokens
        result.append(f"### {title}\n**Source:** {url}\n\n{doc.page_content.strip()[:max_length]}...")
    return "\n\n".join(result)

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text with error handling"""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {str(e)}. Using approximate count.")
        # Fallback to approximate counting (1 token ≈ 4 chars)
        return len(text) // 4

def format_conversation_history(messages, max_tokens=1000):
    """Format conversation history for inclusion in the prompt"""
    formatted_history = []
    for msg in messages:
        role = msg.get("role", "user" if "query" in msg else "assistant")
        content = msg.get("content", msg.get("query", msg.get("response", "")))
        formatted_history.append(f"{role.capitalize()}: {content}")
    
    # Join messages with double newlines for clarity
    history_text = "\n\n".join(formatted_history)
    
    # Truncate if too long to fit in context window
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(history_text)
    if len(tokens) > max_tokens:
        # Keep only the most recent messages that fit within the token limit
        tokens = tokens[-max_tokens:]
        history_text = enc.decode(tokens)
        history_text = "...\n" + history_text  # Indicate truncation
    
    return history_text

# === Retrieval Strategies ===
def fusion_strategy(query, llm):
    """Improved fusion strategy with fewer queries and documents"""
    # Generate query variations - only 2 instead of 3
    fusion_chain = fusion_prompt | llm
    response = fusion_chain.invoke({"question": query})
    variants = [line.strip("- ") for line in response.content.strip().split("\n") if line.strip()][:2]
    variants.insert(0, query)  # Add the original query
    
    # Retrieve documents for each variant, but fewer per variant
    seen = set()
    all_docs = []
    
    for variant in variants:
        for doc in vector_store.similarity_search(variant, k=5):  # Only 5 docs per variant
            hash_ = doc.page_content[:100]
            if hash_ not in seen:
                seen.add(hash_)
                all_docs.append(doc)
    
    return all_docs[:5]  # Return max 5 documents

def simple_strategy(query, llm):
    """Direct retrieval with fewer documents"""
    return vector_store.similarity_search(query, k=5)  

# === Redis Conversation Storage ===
async def get_conversation(conversation_id):
    """Get conversation history from Redis with error handling"""
    if not redis_client:
        logger.warning("Redis client not initialized - returning empty conversation history")
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
    if not redis_client:
        logger.warning("Redis client not initialized - skipping message save")
        return
    
    try:
        # Get existing conversation or create new one
        conversation = await get_conversation(conversation_id)
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = time.time()
        
        # Add new message to conversation
        conversation.append(message)
        
        # Save conversation with TTL
        await redis_client.setex(
            f"conv:{conversation_id}", 
            REDIS_TTL, 
            json.dumps(conversation)
        )
    except Exception as e:
        logger.error(f"Error saving message to conversation: {str(e)}")

# === Cache Helper Functions ===
async def get_cached_response(query: str, model_name: str, strategy: str):
    """Get cached response if available with error handling"""
    if not redis_client:
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
    """Cache response for future use with error handling"""
    if not redis_client:
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

# === Streaming Response Generator ===
async def generate_streaming_response(query_request: QueryRequest, background_tasks: BackgroundTasks) -> AsyncGenerator[str, None]:
    """Generate a streaming response for the query with improved error handling."""
    start_time = time.time()
    
    # Initialize conversation ID if not provided
    conversation_id = query_request.conversation_id or str(uuid.uuid4())
    
    try:
        # Check if model is available
        if query_request.model_name not in AVAILABLE_MODELS:
            error_msg = json.dumps({
                "error": f"Model {query_request.model_name} not available. Available models: {list(AVAILABLE_MODELS.keys())}"
            })
            yield f"data: {error_msg}\n\n"
            return
        
        # Save user query to conversation immediately
        user_message = {
            "role": "user",
            "content": query_request.query,
            "timestamp": time.time()
        }
        await save_message_to_conversation(conversation_id, user_message)
        
        # Initialize LLM with streaming option
        llm = AVAILABLE_MODELS[query_request.model_name](streaming=True)
        llm.temperature = query_request.temperature
        llm.max_tokens = query_request.max_tokens
        
        # Get conversation history if needed
        conversation_history = ""
        if query_request.include_history:
            past_messages = await get_conversation(conversation_id)
            # Only include previous messages, not the current query
            if len(past_messages) > 1:  # More than just the current query
                conversation_history = format_conversation_history(past_messages[:-1])
        
        # Fast-path for simple greetings
        if is_simple_greeting(query_request.query):
            # For simple greetings, generate a direct response without using the LLM
            greeting_response = get_greeting_response(query_request.query)
            
            # Stream the greeting response directly
            yield f"data: {json.dumps({'chunk': greeting_response, 'full': greeting_response})}\n\n"
            
            # Save assistant response to conversation
            assistant_message = {
                "role": "assistant",
                "content": greeting_response,
                "timestamp": time.time()
            }
            await save_message_to_conversation(conversation_id, assistant_message)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Send completion metadata
            completion_data = {
                "done": True,
                "metadata": {
                    "model": "fast-path-greeting",  # Indicate this was handled directly
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
            
        # For non-greeting queries, proceed with normal processing
        # Select retrieval strategy
        retrieve_fn = fusion_strategy if query_request.strategy == "fusion" else simple_strategy
        
        # Retrieve relevant documents (with timeout protection)
        try:
            docs = retrieve_fn(query_request.query, llm)
        except Exception as e:
            logger.warning(f"Error in retrieval: {str(e)}. Falling back to simple strategy.")
            docs = simple_strategy(query_request.query, llm)
            
        # Format documents and create context (with shorter max length for speed)
        context = format_docs(docs, max_length=600)
    
        # Create prompt with history
        prompt = final_prompt.format(
            history=conversation_history,
            context=context, 
            question=query_request.query
        )
        
        # Count tokens
        tokens_used = count_tokens(prompt, query_request.model_name)
        
        # Create chain with callbacks for streaming
        chain = llm | StrOutputParser()
        
        # Stream the response
        full_response = ""
        async for chunk in chain.astream(prompt):
            full_response += chunk
            yield f"data: {json.dumps({'chunk': chunk, 'full': full_response})}\n\n"
        
        # Save assistant response to conversation
        assistant_message = {
            "role": "assistant",
            "content": full_response,
            "timestamp": time.time()
        }
        await save_message_to_conversation(conversation_id, assistant_message)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Create sources list (with shorter snippets)
        sources = [
            {
                "title": doc.metadata.get("title", "Untitled"),
                "url": doc.metadata.get("url", "No URL"),
                "snippet": doc.page_content[:150] + "..."  # Shorter snippets
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
        yield f"data: {json.dumps(error_data)}\n\n"
        
        # Send completion with error info
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

# === Core Query Processing ===
async def process_query(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """Process a query with improved error handling and conversation management"""
    start_time = time.time()
    
    # Initialize conversation ID if not provided
    conversation_id = query_request.conversation_id or str(uuid.uuid4())
    
    try:
        # Save user query to conversation immediately
        user_message = {
            "role": "user",
            "content": query_request.query,
            "timestamp": time.time()
        }
        await save_message_to_conversation(conversation_id, user_message)
        
        # Check cache first for identical queries (if not streaming)
        if not query_request.stream:
            cached = await get_cached_response(
                query_request.query, 
                query_request.model_name,
                query_request.strategy
            )
            if cached:
                # Update conversation ID in cached response
                cached["metadata"]["conversation_id"] = conversation_id
                
                # Save assistant response to conversation
                assistant_message = {
                    "role": "assistant",
                    "content": cached["response"],
                    "timestamp": time.time()
                }
                await save_message_to_conversation(conversation_id, assistant_message)
                
                # Return cached response with updated conversation ID
                return QueryResponse(**cached)
        
        # Get LLM model
        if query_request.model_name not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Model {query_request.model_name} not available. Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        
        # Initialize LLM with streaming option if requested
        llm = AVAILABLE_MODELS[query_request.model_name](streaming=query_request.stream)
        llm.temperature = query_request.temperature
        llm.max_tokens = query_request.max_tokens
        
        # Get conversation history if needed
        conversation_history = ""
        if query_request.include_history:
            past_messages = await get_conversation(conversation_id)
            # Only include previous messages, not the current query
            if len(past_messages) > 1:  # More than just the current query
                conversation_history = format_conversation_history(past_messages[:-1])
        
        # Fast-path for simple greetings
        if is_simple_greeting(query_request.query):
            # For simple greetings, generate a direct response without using the LLM
            greeting_response = get_greeting_response(query_request.query)
            
            # Save assistant response to conversation
            assistant_message = {
                "role": "assistant",
                "content": greeting_response,
                "timestamp": time.time()
            }
            await save_message_to_conversation(conversation_id, assistant_message)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Create response
            response = QueryResponse(
                response=greeting_response,
                metadata=ResponseMetadata(
                    model="fast-path-greeting",  # Indicate this was handled directly
                    strategy="direct",
                    chunks_retrieved=0,
                    tokens_used=0,
                    processing_time=round(duration, 2),
                    conversation_id=conversation_id
                ),
                context_sources=[]
            )
            
            return response
            
        # For non-greeting queries, proceed with normal processing
        # Select retrieval strategy
        retrieve_fn = fusion_strategy if query_request.strategy == "fusion" else simple_strategy
        
        # Retrieve relevant documents (with timeout protection)
        try:
            docs = retrieve_fn(query_request.query, llm)
        except Exception as e:
            logger.warning(f"Error in retrieval: {str(e)}. Falling back to simple strategy.")
            docs = simple_strategy(query_request.query, llm)
        
        # Format documents and create context
        context = format_docs(docs, max_length=600)
        
        # Create prompt with history
        prompt = final_prompt.format(
            history=conversation_history,
            context=context, 
            question=query_request.query
        )
        
        # Count tokens
        tokens_used = count_tokens(prompt, query_request.model_name)
        
        # Generate response
        parser = StrOutputParser()
        answer = (llm | parser).invoke(prompt)
        
        # Save assistant response to conversation
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
                "snippet": doc.page_content[:150] + "..."  # Shorter snippets
            }
            for doc in docs
        ]
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Create response
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
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# === API Endpoints ===
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and available models"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        available_models=list(AVAILABLE_MODELS.keys())
    )

@app.get("/clear-cache")
async def clear_cache():
    """Clear the response cache"""
    if not redis_client:
        raise HTTPException(
            status_code=500,
            detail="Redis client not initialized"
        )
    
    try:
        # Get all cache keys
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
    # Try to get conversation_id from cookie
    conversation_id = request.cookies.get("conversation_id")
    
    if not conversation_id:
        # Generate new ID if none exists
        conversation_id = str(uuid.uuid4())
        logger.info(f"Created new conversation: {conversation_id}")
    
    return conversation_id

@app.post(
    "/query", 
    dependencies=[Depends(RateLimiter(times=20, seconds=60))]  # Rate limit
)
async def query_endpoint(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """Process a legal query using the specified LLM and retrieval strategy"""
    # Auto-generate conversation ID if not provided
    if not query_request.conversation_id:
        query_request.conversation_id = await get_or_create_conversation(request)
    
    # If streaming is requested, return a streaming response
    if query_request.stream:
        response = StreamingResponse(
            generate_streaming_response(query_request, background_tasks),
            media_type="text/event-stream"
        )
        
        # Set conversation ID cookie in response
        response.set_cookie(
            key="conversation_id",
            value=query_request.conversation_id,
            httponly=True,
            max_age=30*24*60*60  # 30 days
        )
        
        return response
    
    # Regular non-streaming response
    try:
        response_data = await process_query(query_request, background_tasks)
        
        # Return the response with cookie
        response = JSONResponse(content=response_data.dict())
        response.set_cookie(
            key="conversation_id",
            value=query_request.conversation_id,
            httponly=True,
            max_age=30*24*60*60  # 30 days
        )
        
        return response
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Log and convert other exceptions to HTTP exceptions
        logger.error(f"Unexpected error in query endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Retrieve conversation history by ID"""
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
    if not redis_client:
        raise HTTPException(
            status_code=500,
            detail="Redis client not initialized"
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
    if not redis_client:
        raise HTTPException(
            status_code=500,
            detail="Redis client not initialized"
        )
    
    try:
        # Get the conversation
        conversation = await get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        # Check if message index is valid
        if message_index < 0 or message_index >= len(conversation):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid message index {message_index}"
            )
        
        # Remove the message
        removed_message = conversation.pop(message_index)
        
        # Save the updated conversation
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
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error deleting message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting message: {str(e)}"
        )

@app.delete("/conversation/{conversation_id}/messages")
async def delete_multiple_messages(conversation_id: str, request: BulkDeleteRequest):
    """Delete multiple messages from a conversation by their indices"""
    if not redis_client:
        raise HTTPException(
            status_code=500,
            detail="Redis client not initialized"
        )
    
    try:
        # Get the conversation
        conversation = await get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        # Sort indices in descending order to avoid index shifting issues
        indices = sorted(request.message_indices, reverse=True)
        
        # Check if indices are valid
        if any(idx < 0 or idx >= len(conversation) for idx in indices):
            raise HTTPException(
                status_code=400,
                detail="One or more invalid message indices"
            )
        
        # Remove messages
        removed = []
        for idx in indices:
            removed.append(conversation.pop(idx))
        
        # Save the updated conversation
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
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error deleting messages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting messages: {str(e)}"
        )
