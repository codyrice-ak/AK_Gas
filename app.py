import streamlit as st
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Alaska Gas Pipeline RAG System",
    page_icon="🛢️",
    layout="wide"
)

@st.cache_data
def load_documents():
    """Load processed documents"""
    try:
        with open('data/processed_documents.pkl', 'rb') as f:
            documents = pickle.load(f)
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return []

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings model"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"}
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return None

def search_documents(query, documents, embeddings, top_k=5):
    """Simple text search through documents"""
    if not embeddings:
        # Fallback to keyword search
        results = []
        query_lower = query.lower()
        
        for doc in documents:
            if any(word in doc.page_content.lower() for word in query_lower.split()):
                results.append(doc)
                if len(results) >= top_k:
                    break
        return results
    
    # TODO: Add vector similarity search in next step
    return documents[:top_k]  # Return first 5 for now

def main():
    st.title("🛢️ Alaska Gas Pipeline Document Query System")
    
    # Load documents
    documents = load_documents()
    if documents:
        st.success(f"✅ Successfully loaded {len(documents):,} document chunks!")
    else:
        st.error("❌ Could not load documents")
        return
    
    # Initialize embeddings
    with st.spinner("Initializing embeddings model..."):
        embeddings = initialize_embeddings()
    
    if embeddings:
        st.success("✅ Embeddings model initialized!")
    else:
        st.warning("⚠️ Embeddings model failed - using keyword search fallback")
    
    # Query interface
    user_query = st.chat_input("Ask your question about Alaska Gas Pipeline documents...")
    
    if user_query:
        st.write(f"**Your question:** {user_query}")
        
        with st.spinner("Searching through documents..."):
            results = search_documents(user_query, documents, embeddings)
        
        if results:
            st.write(f"**Found {len(results)} relevant documents:**")
            
            for i, doc in enumerate(results, 1):
                with st.expander(f"Result {i}: {doc.metadata.get('filename', 'Unknown')}"):
                    st.write(doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content)
                    st.caption(f"Source: {doc.metadata.get('filename', 'Unknown')}")
        else:
            st.warning("No relevant documents found. Try different keywords.")

if __name__ == "__main__":
    main()