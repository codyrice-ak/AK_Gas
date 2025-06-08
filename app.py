import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(
    page_title="Alaska Gas Pipeline RAG System",
    page_icon="🛢️",
    layout="wide"
)

@st.cache_data
def load_embeddings_data():
    """Load precomputed embeddings for semantic search"""
    try:
        with open('data/streamlit_embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None

@st.cache_data
def load_tfidf_data():
    """Load TF-IDF backup for query encoding"""
    try:
        with open('data/tfidf_backup.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading TF-IDF data: {e}")
        return None

def semantic_search(query, embeddings_data, tfidf_data, top_k=15):
    """Perform semantic search using precomputed embeddings"""
    
    if embeddings_data is None or tfidf_data is None:
        return []
    
    try:
        # Use TF-IDF to encode the query
        query_vector = tfidf_data['vectorizer'].transform([query])
        
        # Find most similar documents using TF-IDF
        tfidf_similarities = cosine_similarity(query_vector, tfidf_data['tfidf_matrix']).flatten()
        
        # Get top documents
        top_indices = np.argsort(tfidf_similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if tfidf_similarities[idx] > 0.01:  # Minimum similarity threshold
                results.append({
                    'content': embeddings_data['contents'][idx],
                    'metadata': embeddings_data['metadata'][idx],
                    'similarity': tfidf_similarities[idx]
                })
        
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def extract_financial_info(results, query):
    """Extract financial information from search results"""
    findings = []
    
    # Financial patterns based on query type
    if any(word in query.lower() for word in ['interest', 'rate']):
        patterns = [
            r'interest\s+rate[s]?\s*(?:of|at|is|was)?\s*[\d.]+\s*(?:%|percent)',
            r'[\d.]+\s*(?:%|percent)\s*interest',
            r'financing\s*(?:at|with)?\s*[\d.]+\s*(?:%|percent)',
            r'debt\s*(?:at|with)?\s*[\d.]+\s*(?:%|percent)'
        ]
    else:
        patterns = [
            r'\$[\d,]+(?:\.\d{1,2})?\s*(?:billion|million|thousand)',
            r'[\d,]+(?:\.\d{1,2})?\s*billion\s*(?:dollars?|USD)',
            r'cost.*?\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:billion|million))?'
        ]
    
    year_pattern = r'\b(19\d{2}|20\d{2})\b'
    
    for result in results:
        content = result['content']
        metadata = result['metadata']
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                finding_text = match.group()
                start_pos = max(0, match.start() - 200)
                end_pos = min(len(content), match.end() + 200)
                context = content[start_pos:end_pos].strip()
                
                # Extract year
                years = re.findall(year_pattern, context)
                year = years[0] if years else "Year not specified"
                
                findings.append({
                    'finding': finding_text,
                    'context': context,
                    'filename': metadata.get('filename', 'Unknown'),
                    'year': year,
                    'similarity': result['similarity']
                })
    
    return findings

def main():
    st.title("🛢️ Alaska Gas Pipeline Document Query System")
    
    # Load data
    embeddings_data = load_embeddings_data()
    tfidf_data = load_tfidf_data()
    
    if embeddings_data and tfidf_data:
        st.success(f"✅ Loaded {len(embeddings_data['contents']):,} document embeddings for semantic search!")
        st.info("🔍 Using optimized semantic search with precomputed embeddings")
    else:
        st.error("❌ Could not load embeddings data")
        return
    
    # Query interface
    user_query = st.chat_input("Ask your question about Alaska Gas Pipeline documents...")
    
    if user_query:
        st.write(f"**Your question:** {user_query}")
        
        with st.spinner("Performing semantic search through 39,581 document chunks..."):
            results = semantic_search(user_query, embeddings_data, tfidf_data)
        
        if results:
            # Extract financial information
            financial_findings = extract_financial_info(results, user_query)
            
            if financial_findings and any(word in user_query.lower() for word in ['interest', 'rate', 'cost', 'billion', 'million', '$']):
                st.markdown("## 📊 Financial Information Found\n")
                
                for i, finding in enumerate(financial_findings[:10], 1):
                    st.markdown(f"**{i}. {finding['finding']}** [{i}]")
                    st.markdown(f"   - **Source:** {finding['filename']}")
                    st.markdown(f"   - **Year:** {finding['year']}")
                    st.markdown(f"   - **Similarity:** {finding['similarity']:.3f}")
                    st.markdown(f"   - **Context:** {finding['context'][:200]}...\n")
                
                # Add footnotes
                st.markdown("---\n### 📚 Sources:")
                for i, finding in enumerate(financial_findings[:10], 1):
                    st.markdown(f"[{i}] {finding['filename']} ({finding['year']})")
            else:
                st.write("**📋 Relevant Documents Found:**")
                for i, result in enumerate(results[:5], 1):
                    with st.expander(f"Result {i}: {result['metadata'].get('filename', 'Unknown')} (Similarity: {result['similarity']:.3f})"):
                        st.write(result['content'])
                        st.caption(f"Source: {result['metadata'].get('filename', 'Unknown')}")
        else:
            st.warning("No relevant documents found. Try rephrasing your question.")

if __name__ == "__main__":
    main()
