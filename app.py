import streamlit as st
import pickle
import re

st.set_page_config(
    page_title="Alaska Gas Pipeline RAG System",
    page_icon="🛢️",
    layout="wide"
)

def main():
    st.title("🛢️ Alaska Gas Pipeline Document Query System")
    
    try:
        # Test basic functionality
        st.success("✅ App is running successfully!")
        
        # Test file access
        try:
            with open('data/processed_documents.pkl', 'rb') as f:
                documents = pickle.load(f)
            st.success(f"✅ Successfully loaded {len(documents):,} document chunks!")
            
            # Simple search interface
            user_query = st.text_input("Enter your question about Alaska Gas Pipeline documents:")
            
            if user_query:
                st.write(f"**Your question:** {user_query}")
                
                # Simple keyword search
                query_words = user_query.lower().split()
                relevant_docs = []
                
                for doc in documents[:100]:  # Limit search for performance
                    content_lower = doc.page_content.lower()
                    score = sum(1 for word in query_words if word in content_lower)
                    
                    if score > 0:
                        relevant_docs.append((doc, score))
                
                # Sort and display results
                relevant_docs.sort(key=lambda x: x[1], reverse=True)
                
                if relevant_docs:
                    st.write(f"**Found {len(relevant_docs)} relevant documents:**")
                    
                    for i, (doc, score) in enumerate(relevant_docs[:5], 1):
                        with st.expander(f"Result {i}: {doc.metadata.get('filename', 'Unknown')} (Score: {score})"):
                            st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.caption(f"Source: {doc.metadata.get('filename', 'Unknown')}")
                else:
                    st.warning("No relevant documents found. Try different keywords.")
                    
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            st.info("Check if your data files are properly uploaded to GitHub.")
            
    except Exception as e:
        st.error(f"Critical error: {e}")
        st.error("App failed to initialize properly.")

if __name__ == "__main__":
    main()