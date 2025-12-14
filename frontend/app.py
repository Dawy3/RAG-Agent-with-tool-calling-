# streamlit_intelligent_rag.py
import streamlit as st
import requests
import time
import os
from datetime import datetime

st.set_page_config(
    page_title="Intelligent RAG Agent",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .tool-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    .kb-badge {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .web-badge {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    .calc-badge {
        background-color: #e8f5e9;
        color: #388e3c;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Sidebar
with st.sidebar:
    st.title("🧠 Agent Control Panel")
    
    # Upload documents
    st.subheader("📄 Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt', 'docx'],
        help="Upload documents to expand the knowledge base"
    )
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            files = {'file': uploaded_file}
            response = requests.post(f"{API_URL}/documents/upload", files=files)
            if response.status_code == 200:
                st.success("✅ Document added!")
    
    st.divider()
    
    # Agent Analytics
    st.subheader("📊 Agent Analytics")
    
    try:
        analytics = requests.get(f"{API_URL}/agent/analytics").json()
        
        st.metric("Total Queries", analytics['total_queries'])
        st.metric("Avg Tools/Query", analytics['avg_tools_per_query'])
        
        st.write("**Tool Usage:**")
        for tool_stat in analytics['tool_usage']:
            st.write(f"• {tool_stat['tool']}: {tool_stat['count']}")
    except:
        st.warning("Analytics not available")
    
    st.divider()
    
    # Agent capabilities
    with st.expander("🛠️ Agent Capabilities"):
        st.write("""
        **Available Tools:**
        - 🗄️ Knowledge Base Search
        - 🌐 Web Search (Tavily)
        - 🧮 Calculator
        
        **Intelligence:**
        - Automatic tool selection
        - Multi-step reasoning
        - Source tracking
        - Context awareness
        """)

# Main area
st.title("🧠 Intelligent RAG Agent")
st.caption("Ask anything - the agent will intelligently choose the right tools")

# Example queries
st.subheader("💡 Example Queries")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📚 Query Knowledge Base"):
        st.session_state.example_query = "What information do we have about machine learning?"

with col2:
    if st.button("🌐 Current Events"):
        st.session_state.example_query = "What are the latest developments in AI?"

with col3:
    if st.button("🧮 Math & Analysis"):
        st.session_state.example_query = "Calculate the compound interest on $10,000 at 5% for 10 years"

# Query input
query = st.text_area(
    "Your question:",
    value=st.session_state.get('example_query', ''),
    placeholder="Ask me anything...",
    height=100
)

if st.button("🚀 Ask Agent", type="primary", use_container_width=True):
    if query:
        with st.spinner("🤔 Agent is thinking and selecting tools..."):
            start_time = time.time()
            
            response = requests.post(
                f"{API_URL}/agent/query",
                json={"query": query, "session_id": "streamlit-session"}
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Display answer
                st.success("**Answer:**")
                st.markdown(result['answer'])
                
                # Display metadata
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Response Time", f"{elapsed:.2f}s")
                
                with col2:
                    st.metric("Reasoning Steps", result['reasoning_steps'])
                
                with col3:
                    st.metric("Tools Used", len(result['tool_used']))
                
                # Display tools used with badges
                if result['tool_used']:
                    st.write("**🛠️ Tools Used:**")
                    tools_html = ""
                    for tool in result['tool_used']:
                        badge_class = {
                            'search_knowledge_base': 'kb-badge',
                            'search_web': 'web-badge',
                            'calculate': 'calc-badge'
                        }.get(tool, 'kb-badge')
                        
                        tool_display = {
                            'search_knowledge_base': '🗄️ Knowledge Base',
                            'search_web': '🌐 Web Search',
                            'calculate': '🧮 Calculator'
                        }.get(tool, tool)
                        
                        tools_html += f'<span class="tool-badge {badge_class}">{tool_display}</span>'
                    
                    st.markdown(tools_html, unsafe_allow_html=True)
                
                # Show sources if available
                if result.get('sources'):
                    with st.expander("📚 Sources"):
                        for source in result['sources']:
                            st.write(f"• {source}")
            else:
                st.error("❌ Query failed")
    else:
        st.warning("Please enter a question")

# Show recent queries
st.divider()
st.subheader("📝 Recent Queries")

# Mock recent queries (implement actual history)
recent = [
    {"query": "What is in our documentation about APIs?", "tools": ["🗄️"], "time": "2 min ago"},
    {"query": "Latest AI news today", "tools": ["🌐"], "time": "5 min ago"},
    {"query": "Calculate ROI for our project", "tools": ["🧮"], "time": "10 min ago"},
]

for item in recent:
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.write(f"**{item['query']}**")
        col2.write(" ".join(item['tools']))
        col3.write(item['time'])