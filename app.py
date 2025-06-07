import streamlit as st
import pickle
import re
from pathlib import Path
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def extract_cost_information(docs, query):
    """Extract specific cost information from retrieved documents"""
    cost_examples = []
    
    # Patterns to find costs, years, and page numbers
    cost_pattern = r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|M|B|K))?'
    year_pattern = r'\b(19|20)\d{2}\b'
    page_pattern = r'(?:page|p\.?)\s*(\d+)'
    
    for i, doc in enumerate(docs):
        content = doc.page_content
        metadata = doc.metadata
        
        # Find costs in the document
        costs = re.findall(cost_pattern, content, re.IGNORECASE)
        years = re.findall(year_pattern, content)
        pages = re.findall(page_pattern, content, re.IGNORECASE)
        
        if costs:
            # Extract surrounding context for each cost
            for cost in costs[:3]:  # Limit to 3 costs per document
                # Find sentence containing the cost
                sentences = content.split('.')
                cost_sentence = ""
                for sentence in sentences:
                    if cost in sentence:
                        cost_sentence = sentence.strip()
                        break
                
                # Get document info
                filename = metadata.get('filename', 'Unknown')
                source = metadata.get('source', 'Unknown')
                
                # Try to find year and page in the same context
                context_year = years[0] if years else "Year not specified"
                context_page = pages[0] if pages else "Page not specified"
                
                cost_examples.append({
                    'cost': cost,
                    'context': cost_sentence,
                    'filename': filename,
                    'year': context_year,
                    'page': context_page,
                    'source': source
                })
    
    return cost_examples

def generate_structured_answer(docs, query):
    """Generate a structured answer from retrieved documents"""
    
    if "cost" in query.lower() and "example" in query.lower():
        cost_info = extract_cost_information(docs, query)
        
        if cost_info:
            answer = "## 💰 Cost Examples Found in Alaska Gas Pipeline Documents\n\n"
            
            for i, item in enumerate(cost_info[:10], 1):  # Limit to 10 examples
                answer += f"### Example {i}\n"
                answer += f"**Cost:** {item['cost']}\n"
                answer += f"**Context:** {item['context']}\n"
                answer += f"**Document:** {item['filename']}\n"
                answer += f"**Year:** {item['year']}\n"
                answer += f"**Page:** {item['page']}\n"
                answer += f"**Source:** {item['source']}\n\n"
                answer += "---\n\n"
            
            return answer
        else:
            return "No specific cost information found in the retrieved documents."
    
    else:
        # For general queries, provide a summary
        answer = "## 📋 Summary from Alaska Gas Pipeline Documents\n\n"
        
        for i, doc in enumerate(docs[:3], 1):
            answer += f"### Source {i}: {doc.metadata.get('filename', 'Unknown')}\n"
            answer += f"{doc.page_content[:500]}...\n\n"
        
        return answer

# Update your main query processing section
def process_query(user_query, vector_store):
    """Process user query and generate structured response"""
    try:
        with st.spinner("Searching through 39,581 document chunks..."):
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 10}  # Get more documents for better analysis
            )
            docs = retriever.get_relevant_documents(user_query)
        
        if docs:
            # Generate structured answer
            structured_answer = generate_structured_answer(docs, user_query)
            
            # Display the answer
            st.markdown(structured_answer)
            
            # Also show raw documents in an expander
            with st.expander("📄 View Raw Document Chunks"):
                for i, doc in enumerate(docs, 1):
                    st.write(f"**Document {i}: {doc.metadata.get('filename', 'Unknown')}**")
                    st.write(doc.page_content)
                    st.write("---")
        else:
            st.warning("No relevant information found. Try rephrasing your question.")
            
    except Exception as e:
        st.error(f"Search error: {e}")

# Update your main function's query section
def main():
    st.title("🛢️ Alaska Gas Pipeline Document Query System")
    
    # ... (your existing loading code) ...
    
    # Query interface
    user_query = st.chat_input("Ask your question about Alaska Gas Pipeline documents...")
    
    if user_query and st.session_state.vector_store:
        process_query(user_query, st.session_state.vector_store)