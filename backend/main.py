from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, UploadFile, File,  HTTPException
from pydantic import BaseModel

from typing import TypedDict, Literal
import asyncpg
import os 
import uuid
import tempfile

app = FastAPI(title="Intelligent RAG Agent")
# Openrouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = os.getenv("MODEL_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME= "intelligent-rag"

# Initialize sentence-transformers embedding
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}
)
sample_embedding = embeddings.embed_query("test")
DIMENSION = len(sample_embedding)

# Create index if it doens't exists 
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name= INDEX_NAME,
        dimension= DIMENSION,
        metric= "cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Initialize vector store
vectorstore= PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# Define custom tools
@tool
async def search_knowledge_base(query:str) -> str:
    """Search The internal knowledge base for relevant documetns.
    Use this when the user ask about documents you have or internal information."""
    
    docs = await vectorstore.asimilarity_search(query, k=4)
    
    if not docs:
        return "No relevant documents found in knowledge base."
    
    results = "\n\n".join([
        f"Document {i+1} (from {doc.metadata.get('filename','unknown')}):\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])
    
    return f"Knowledge Base Results:\n{results}"

@tool
async def search_web(query:str)-> str:
    """search the web for current information, news, of facts not in the knowledge base.
    Use this for recent events, real-time data, or when knowledge base has no results."""
    
    search_tool = TavilySearchResults(
        max_results=3,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True
    )
    
    results= await search_tool.ainvoke({"query": query})
    
    formatted = []
    for r in results:
        formatted.append(
            f"Source: {r.get('url','N/A')}\n"
            f"Title: {r.get('title','N/A')}\n"
            f"Content: {r.get('content','N/A')}\n"
        )
    
    return "Web Search Results:\n" + "\n---\n".join(formatted)

@tool
async def calculate(expression:str) -> str:
    """Perform mathmatical calculations.
    Use this for any math operations or numerical computations."""
    
    try:
        # Safe eval for basic math
        result = eval(expression, {"__builtins__":{}}, {})
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"
    
    
# Create tools list
tools = [search_knowledge_base, search_web, calculate]
tool_node = ToolNode(tools)

# Agent State
class AgentState(TypedDict):
    messages: list
    tool_calls_made: int
    sources_used: list
    
# Build the agent graph
def create_intelligent_agent():
    """Create an agent that intelligently decides which tools to use"""
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key= OPENROUTER_API_KEY,
        base_url= OPENROUTER_BASE_URL,
        temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    # Agent node - decides what to do
    async def agent(state:AgentState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return{"messages": [response]}
    
    # Router - decides wether to use tools or finish
    def should_continue(state:AgentState) -> Literal["tools","end"]:
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # otherwise, end
        return "end"
    
    
    # Build Graph
    graph = StateGraph(AgentState)
    
    # Add Nodes
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    
    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    graph.add_edge("tools", "agent") #  After tools, go back to agent 
    
    return graph.compile()

agent_graph = create_intelligent_agent()

class QueryRequest(BaseModel):
    query:str
    session_id:str = "default"
    
class QueryResponse(BaseModel):  
    query: str
    answer: str
    tool_used: list[str]    
    sources : list[str]
    reasoning_steps: int
    
@app.post("/agent/query", response_model= QueryResponse)
async def query_agent(request: QueryRequest):
    """Query the intelligent RAG agent"""
    
    try:
        # Prepare initial state
        initial_state = {
            "messages": [
                SystemMessage(content="""You are intelligent assistant with access to:
                              1. Internal knowledge base (document uploaded by users)
                              2. Web search (for current information)
                              3. Calculator (for math)
                              IMPORTATN:
                              - ALWAYS search the knowledge base FIRST for any document-related questions
                              - Use web search for current events, news, or when knowledge base has no results
                              - Use multiple tools if needed
                              - Provide comprehensive answers with source citations"""),
                HumanMessage(content=request.query)
            ],
            "tool_calls_made": 0,
            "sources_used": []
        }
        
        # Run the agent
        result = await agent_graph.ainvoke(initial_state)
        
        # Extract answer and metadata
        messages = result["messages"]
        final_answer = messages[-1].content
        
        # Track which tools were used
        tool_used = []
        sources = []
        
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    if tool_name not in tool_used:
                        tool_used.append(tool_name)
                        
        return QueryResponse(
            query= request.query,
            answer= final_answer,
            tool_used= tool_used,
            sources= sources,
            reasoning_steps = len(messages) // 2
            
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load document
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata
        doc_id = str(uuid.uuid4())
        for chunk in chunks:
            chunk.metadata.update({
                "doc_id": doc_id,
                "filename": file.filename
            })
        
        # Store in Pinecone (async)
        await vectorstore.aadd_documents(chunks)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks_created": len(chunks),
            "status": "success"
        }
    
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/agent/analytics")
async def get_analytics():
    """Get agent usage analytics"""
    
    # connect to PostgreSQL
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    
    try:
        # Implement analytics queries
        total_queries  = await conn.fetchval(
            "SELECT COUNT(*) FROM agent_queries"
        )

        avg_tools_per_query = await conn.fetchval(
            "SELECT AVG(tools_used) FROM agent_queries"
        )
        
        tool_usage = await conn.fetch(
            """SELECT tool_name, COUNT(*) as usage_count 
               FROM agent_tool_usage 
               GROUP BY tool_name 
               ORDER BY usage_count DESC"""
        )
        return {
            "total_queries": total_queries,
            "avg_tools_per_query": round(avg_tools_per_query, 2),
            "tool_usage": [
                {"tool": row["tool_name"], "count": row["usage_count"]}
                for row in tool_usage
            ]
        }
    finally:
        await conn.close()
        
        
        
        