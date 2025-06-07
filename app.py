import streamlit as st
import pickle
from pathlib import Path
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

st.set_page_config(
    page_title="Alaska Gas Pipeline RAG System",
    page_icon="🛢️",
    layout="wide"
)

@st.cache_resource
def load_precomputed_embeddings():
    """Load precomputed embeddings from your processed files"""
    try:
        # Load your precomputed data
        with open('data/cloud_embeddings.pkl', 'rb') as f:
            points_data = pickle.load(f)
        
        with open('data/processed_documents.pkl', 'rb') as f:
            documents = pickle.load(f)
        
        # Initialize embeddings model (same as used for processing)
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"}
        )
        
        # Create in-memory Qdrant client
        client = QdrantClient(":memory:")
        
        # Create collection
        client.create_collection(
            collection_name="rag_collection",
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )
        
        # Restore points to Qdrant
        for point in points_data:
            client.upsert(
                collection_name="rag_collection",
                points=[point]
            )
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="rag_collection",
            embedding=embeddings
        )
        
        return vector_store, len(documents)
        
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None, 0

def main():
    st.title("🛢️ Alaska Gas Pipeline Document Query System")
    
    # Load precomputed embeddings
    if 'vector_store' not in st.session_state:
        with st.spinner("Loading precomputed embeddings..."):
            st.session_state.vector_store, doc_count = load_precomputed_embeddings()
        
        if st.session_state.vector_store:
            st.success(f"✅ Loaded {doc_count:,} document chunks ready for search!")
        else:
            st.error("❌ Failed to load embeddings")
            return
    
    # Query interface
    user_query = st.chat_input("Ask your question about Alaska Gas Pipeline documents...")
    
    if user_query and st.session_state.vector_store:
        try:
            with st.spinner("Searching through 39,581 document chunks..."):
                # Search through your precomputed embeddings
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 5}
                )
                docs = retriever.get_relevant_documents(user_query)
            
            if docs:
                st.write("**📋 Relevant Information Found:**")
                
                for i, doc in enumerate(docs, 1):
                    with st.expander(f"Result {i} - {doc.metadata.get('filename', 'Unknown')}"):
                        st.write(doc.page_content)
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
            else:
                st.warning("No relevant information found. Try rephrasing your question.")
                
        except Exception as e:
            st.error(f"Search error: {e}")
    
    # Sidebar info
    with st.sidebar:
        st.subheader("📊 System Status")
        if st.session_state.get('vector_store'):
            st.success("✅ 39,581 document chunks loaded")
            st.info("🔍 Ready for queries")
        else:
            st.error("❌ Embeddings not loaded")

if __name__ == "__main__":
    main()
