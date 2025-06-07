import streamlit as st

st.title("🛢️ Alaska Gas Pipeline Document Query System")

try:
    # Test if files exist
    import os
    if os.path.exists('data/cloud_embeddings.pkl'):
        st.success("✅ Embeddings file found")
    else:
        st.error("❌ Embeddings file not found")
        
    if os.path.exists('data/processed_documents.pkl'):
        st.success("✅ Documents file found")
    else:
        st.error("❌ Documents file not found")
        
    st.info("App is working - file loading temporarily disabled for debugging")
    
except Exception as e:
    st.error(f"Error: {e}")

# Simple query interface for testing
user_query = st.chat_input("Test query (functionality disabled for debugging)...")
if user_query:
    st.write(f"You asked: {user_query}")
    st.info("Query processing temporarily disabled - working on loading embeddings")