import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import tempfile
import os

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings model - cached to avoid recomputation"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def initialize_vector_store():
    """Initialize Qdrant vector store - works better on Streamlit Cloud"""
    embeddings = initialize_embeddings()
    
    # Use in-memory Qdrant for Streamlit Cloud
    client = QdrantClient(":memory:")
    
    # Create collection
    collection_name = "rag_collection"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,  # BAAI/bge-small-en-v1.5 dimension
            distance=Distance.COSINE
        )
    )
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

def main():
    st.title("🤖 RAG Query System")
    
    try:
        if 'vector_store' not in st.session_state:
            with st.spinner("Initializing system..."):
                st.session_state.vector_store = initialize_vector_store()
                st.session_state.retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
        
        user_query = st.chat_input("Ask your question...")
        
        if user_query:
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
