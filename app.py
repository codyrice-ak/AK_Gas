import streamlit as st
import pickle
import re
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

def extract_year_from_context(context):
    """Enhanced year extraction with multiple patterns"""
    year_patterns = [
        r'\b(19\d{2}|20\d{2})\b',  # Standard 4-digit years
        r'\b(19\d{2}|20\d{2})\s*dollars?\b',  # Years followed by "dollars"
        r'\bin\s+(19\d{2}|20\d{2})\b',  # "in 1977", "in 2019"
        r'\b(19\d{2}|20\d{2})\s+estimate\b'  # "1977 estimate"
    ]
    
    for pattern in year_patterns:
        matches = re.findall(pattern, context, re.IGNORECASE)
        if matches:
            # Return the first valid 4-digit year found
            for match in matches:
                if isinstance(match, tuple):
                    year = match[0] if len(match[0]) == 4 else match[1]
                else:
                    year = match
                if len(year) == 4:
                    return year
    
    return "Year not specified"

def extract_cost_estimates(documents):
    """Extract cost estimates from documents with context"""
    cost_estimates = []
    
    # Enhanced patterns for cost detection
    cost_patterns = [
        r'\$[\d,]+(?:\.\d{1,2})?\s*(?:billion|million|thousand|B|M|K)',
        r'[\d,]+(?:\.\d{1,2})?\s*billion\s*(?:dollars?|USD|\$)',
        r'[\d,]+(?:\.\d{1,2})?\s*million\s*(?:dollars?|USD|\$)',
        r'cost.*?\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:billion|million|thousand))?',
        r'estimate.*?\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:billion|million|thousand))?'
    ]
    
    for doc in documents:
        content = doc.page_content
        metadata = doc.metadata
        
        # Find all cost mentions
        for pattern in cost_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                cost_text = match.group()
                start_pos = max(0, match.start() - 200)
                end_pos = min(len(content), match.end() + 200)
                context = content[start_pos:end_pos].strip()
                
                # Extract year from context - now gets full 4-digit year
                year = extract_year_from_context(context)
                
                cost_estimates.append({
                    'cost': cost_text,
                    'context': context,
                    'filename': metadata.get('filename', 'Unknown'),
                    'year': year,  # Now will show full year
                    'source': metadata.get('source', 'Unknown')
                })
    
    return cost_estimates

def generate_cost_answer(cost_estimates, query):
    """Generate a structured answer about costs"""
    if not cost_estimates:
        return "No cost estimates found in the documents."
    
    # Group and analyze costs
    billion_costs = []
    million_costs = []
    other_costs = []
    
    for estimate in cost_estimates:
        cost_text = estimate['cost'].lower()
        if 'billion' in cost_text:
            billion_costs.append(estimate)
        elif 'million' in cost_text:
            million_costs.append(estimate)
        else:
            other_costs.append(estimate)
    
    # Generate summary
    answer = "## 💰 Cost Estimates for Alaska Gas Pipeline\n\n"
    
    if billion_costs:
        answer += f"**Found {len(billion_costs)} estimates in billions of dollars:**\n\n"
        
        # Extract numerical values for range analysis
        billion_values = []
        for est in billion_costs:
            numbers = re.findall(r'[\d,]+(?:\.\d{1,2})?', est['cost'])
            if numbers:
                try:
                    value = float(numbers[0].replace(',', ''))
                    billion_values.append(value)
                except:
                    pass
        
        if billion_values:
            min_val = min(billion_values)
            max_val = max(billion_values)
            answer += f"**Range: ${min_val:.1f} - ${max_val:.1f} billion USD**\n\n"
    
    # Add detailed estimates
    answer += "### 📋 Detailed Cost Estimates:\n\n"
    
    all_estimates = billion_costs + million_costs + other_costs
    for i, estimate in enumerate(all_estimates[:10], 1):  # Limit to 10
        answer += f"**{i}. {estimate['cost']}**\n"
        answer += f"   - **Source:** {estimate['filename']}\n"
        answer += f"   - **Year:** {estimate['year']}\n"
        answer += f"   - **Context:** {estimate['context'][:200]}...\n\n"
    
    return answer

def search_documents(query, documents, embeddings, top_k=10):
    """Enhanced search with keyword and semantic matching"""
    if not documents:
        return []
    
    # Keyword-based filtering
    query_words = query.lower().split()
    relevant_docs = []
    
    for doc in documents:
        content_lower = doc.page_content.lower()
        
        # Check for cost-related keywords
        cost_keywords = ['cost', 'price', 'estimate', 'billion', 'million', 'dollar', '$', 'budget', 'expense']
        pipeline_keywords = ['pipeline', 'gas', 'alaska', 'transport', 'lng']
        
        cost_score = sum(1 for word in cost_keywords if word in content_lower)
        pipeline_score = sum(1 for word in pipeline_keywords if word in content_lower)
        query_score = sum(1 for word in query_words if word in content_lower)
        
        total_score = cost_score + pipeline_score + query_score
        
        if total_score > 0:
            relevant_docs.append((doc, total_score))
    
    # Sort by relevance score
    relevant_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in relevant_docs[:top_k]]

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
        st.warning("⚠️ Embeddings model failed - using keyword search")
    
    # Query interface
    user_query = st.chat_input("Ask your question about Alaska Gas Pipeline documents...")
    
    if user_query:
        st.write(f"**Your question:** {user_query}")
        
        with st.spinner("Analyzing documents and generating answer..."):
            # Search for relevant documents
            results = search_documents(user_query, documents, embeddings)
            
            # Check if this is a cost-related query
            if any(word in user_query.lower() for word in ['cost', 'price', 'billion', 'million', 'dollar', '$', 'estimate']):
                # Extract cost information
                cost_estimates = extract_cost_estimates(results)
                
                # Generate structured answer
                answer = generate_cost_answer(cost_estimates, user_query)
                st.markdown(answer)
                
            else:
                # For non-cost queries, provide document summaries
                if results:
                    st.write("**📋 Answer based on document analysis:**")
                    
                    # Create a summary answer
                    summary_points = []
                    for doc in results[:5]:
                        content = doc.page_content[:300] + "..."
                        filename = doc.metadata.get('filename', 'Unknown')
                        summary_points.append(f"• **{filename}:** {content}")
                    
                    for point in summary_points:
                        st.markdown(point)
                else:
                    st.warning("No relevant documents found.")
        
        # Show raw documents in expander
        if results:
            with st.expander(f"📄 View {len(results)} Source Documents"):
                for i, doc in enumerate(results, 1):
                    st.write(f"**Document {i}: {doc.metadata.get('filename', 'Unknown')}**")
                    st.write(doc.page_content)
                    st.write("---")

if __name__ == "__main__":
    main()