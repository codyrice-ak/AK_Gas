import streamlit as st
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Alaska Gas Pipeline RAG System",
    page_icon="🛢️",
    layout="wide"
)

@st.cache_data
def load_document_count():
    """Load just the document count for testing"""
    try:
        with open('data/processed_documents.pkl', 'rb') as f:
            documents = pickle.load(f)
        return len(documents)
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return 0

def main():
    st.title("🛢️ Alaska Gas Pipeline Document Query System")
    
    # Test file existence
    if Path('data/cloud_embeddings.pkl').exists():
        st.success("✅ Embeddings file found")
    else:
        st.error("❌ Embeddings file not found")
        
    if Path('data/processed_documents.pkl').exists():
        st.success("✅ Documents file found")
        
        # Load document count
        doc_count = load_document_count()
        if doc_count > 0:
            st.success(f"✅ Successfully loaded {doc_count:,} document chunks!")
        else:
            st.error("❌ Could not load document count")
    else:
        st.error("❌ Documents file not found")
    
    # Simple query interface
    user_query = st.chat_input("Enter your question about Alaska Gas Pipeline documents...")
    if user_query:
        st.write(f"**Your question:** {user_query}")
        st.info("🔧 Query processing will be added in the next step")

if __name__ == "__main__":
    main()