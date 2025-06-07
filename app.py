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
def initialize_vector_store():
    """Initialize vector store with proper persistence handling for Streamlit Cloud"""
    
    # Create a temporary directory for ChromaDB
    if 'STREAMLIT_SHARING_MODE' in os.environ:
        # Running on Streamlit Cloud
        persist_directory = tempfile.mkdtemp()
        st.info("Running on Streamlit Cloud - vector store will be rebuilt on each session")
    else:
        # Running locally
        persist_directory = "./chroma_db"
    
    # Initialize embeddings and vector store
    embeddings = initialize_embeddings()  # Your existing function
    
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name="rag_collection"
    )
    
    return vectorstore


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
