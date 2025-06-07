import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama  # Replace with cloud-compatible LLM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Query System",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings model - cached to avoid recomputation"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}  # Force CPU for cloud deployment
    )

@st.cache_resource
def initialize_vector_store():
    """Initialize vector store - cached to persist across queries"""
    embeddings = initialize_embeddings()
    
    # Use temporary directory for cloud deployment
    persist_directory = tempfile.mkdtemp()
    
    return Chroma(
        collection_name="rag_collection",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

def main():
    st.title("🤖 RAG Query System")
    
    # Initialize components with proper error handling
    try:
        if 'vector_store' not in st.session_state:
            with st.spinner("Initializing system..."):
                st.session_state.vector_store = initialize_vector_store()
                st.session_state.retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
        
        # Your chat interface here
        user_query = st.chat_input("Ask your question...")
        
        if user_query:
            # Process query with error handling
            try:
                # Your RAG logic here
                pass
            except Exception as e:
                st.error(f"Query processing error: {str(e)}")
                
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        st.info("Please refresh the page to retry.")

if __name__ == "__main__":
    main()
