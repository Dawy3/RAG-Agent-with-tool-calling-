# Intelligent RAG Agent with Tool Calling

An intelligent Retrieval-Augmented Generation (RAG) agent that combines document knowledge bases with real-time web search and calculation capabilities. Built with FastAPI backend, Streamlit frontend, and powered by LangGraph for agent orchestration.

## 🚀 Features

- **Multi-Tool Agent**: Integrates knowledge base search, web search, and mathematical calculations
- **Document Upload**: Upload and process PDF documents for knowledge base expansion
- **Real-time Web Search**: Access current information using Tavily Search API
- **Vector Storage**: Uses Pinecone for efficient document embeddings and retrieval
- **Checkpointing**: PostgreSQL-based conversation state persistence
- **Modern UI**: Streamlit-based interface with real-time chat and analytics
- **Docker Support**: Complete containerized deployment with docker-compose

## 🛠️ Tech Stack

- **Backend**: FastAPI, LangGraph, LangChain
- **Frontend**: Streamlit
- **AI/ML**: OpenRouter API, HuggingFace Embeddings
- **Vector DB**: Pinecone
- **Database**: PostgreSQL
- **Search**: Tavily Search API
- **Deployment**: Docker & Docker Compose

## 📋 Prerequisites

- Docker and Docker Compose
- OpenRouter API key
- Pinecone API key
- Tavily Search API key (optional, for web search)

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "RAG Agent with Tool Calling"
```

### 2. Environment Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```env
# OpenRouter API Key for accessing AI models
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Model name to use (e.g., gpt-3.5-turbo, claude-3-haiku, etc.)
MODEL_NAME=gpt-3.5-turbo

# Pinecone API Key for vector database
PINECONE_API_KEY=your_pinecone_api_key_here

# PostgreSQL Database Configuration
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=rag_db
```

### 3. Launch with Docker

Build and start all services:

```bash
docker-compose up --build
```

The application will be available at:
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **Database**: localhost:5432

### 4. Alternative: Local Development

If you prefer running without Docker:

1. **Backend Setup**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Frontend Setup**:
   ```bash
   cd frontend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. **Database**: Set up PostgreSQL locally and update `DATABASE_URL` in `.env`

## 📖 Usage

### Chat Interface

1. Open the Streamlit frontend at http://localhost:8501
2. Enter your query in the chat input
3. The agent will automatically select and use appropriate tools:
   - **Knowledge Base Search** (🗄️): Searches uploaded documents
   - **Web Search** (🌐): Fetches real-time information
   - **Calculator** (🧮): Performs mathematical computations

### Document Upload

1. Use the sidebar "Document Upload" section
2. Select and upload PDF files
3. Documents are automatically processed and added to the knowledge base

### Analytics Dashboard

View conversation statistics and agent performance metrics in the sidebar.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │   PostgreSQL    │
│   Frontend      │◄──►│    Backend      │◄──►│  Checkpointing  │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LangGraph     │    │    Pinecone     │    │   OpenRouter    │
│   Agent Flow    │◄──►│  Vector Store   │    │     Models      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Knowledge      │    │   Web Search    │    │   Calculator    │
│   Base Tool     │    │     Tool        │    │     Tool        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔍 API Endpoints

### Backend API (FastAPI)

- `POST /query` - Query the agent
- `POST /upload` - Upload documents
- `GET /analytics` - Get usage analytics
- `GET /docs` - Interactive API documentation

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | API key for OpenRouter | Yes |
| `MODEL_NAME` | AI model to use | Yes |
| `PINECONE_API_KEY` | API key for Pinecone | Yes |
| `POSTGRES_USER` | PostgreSQL username | Yes |
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes |
| `POSTGRES_DB` | PostgreSQL database name | Yes |
| `API_URL` | Backend API URL (frontend only) | No |

## 🐳 Docker Services

- **backend**: FastAPI application server
- **frontend**: Streamlit web interface
- **db**: PostgreSQL database for checkpointing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [Pinecone](https://pinecone.io) for vector database
- [OpenRouter](https://openrouter.ai) for AI model access
- [Tavily](https://tavily.com) for web search API</content>
<parameter name="filePath">c:\Users\Eldawy\projects\Projects\RAG Agent with Tool Calling\README.md