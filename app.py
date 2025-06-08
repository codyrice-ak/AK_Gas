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

def extract_interest_rates(documents):
    """Extract interest rate information with citations"""
    interest_findings = []
    
    # Enhanced patterns for interest rate detection
    interest_patterns = [
        r'interest\s+rate[s]?\s*(?:of|at|is|was|are|were)?\s*[\d.]+\s*(?:%|percent)',
        r'[\d.]+\s*(?:%|percent)\s*interest',
        r'financing\s*(?:at|with)?\s*[\d.]+\s*(?:%|percent)',
        r'debt\s*(?:at|with)?\s*[\d.]+\s*(?:%|percent)',
        r'borrowing\s*(?:at|with)?\s*[\d.]+\s*(?:%|percent)',
        r'loan\s*(?:at|with)?\s*[\d.]+\s*(?:%|percent)',
        r'cost\s+of\s+capital\s*[\d.]+\s*(?:%|percent)',
        r'discount\s+rate\s*[\d.]+\s*(?:%|percent)'
    ]
    
    year_pattern = r'\b(19\d{2}|20\d{2})\b'
    
    for i, doc in enumerate(documents):
        content = doc.page_content
        metadata = doc.metadata
        
        # Find all interest rate mentions
        for pattern in interest_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                rate_text = match.group()
                start_pos = max(0, match.start() - 300)
                end_pos = min(len(content), match.end() + 300)
                context = content[start_pos:end_pos].strip()
                
                # Extract numerical rate
                rate_numbers = re.findall(r'[\d.]+', rate_text)
                rate_value = rate_numbers[0] if rate_numbers else "N/A"
                
                # Extract year from context
                years = re.findall(year_pattern, context)
                year = years[0] if years else "Year not specified"
                
                interest_findings.append({
                    'rate': rate_text,
                    'rate_value': rate_value,
                    'context': context,
                    'filename': metadata.get('filename', 'Unknown'),
                    'year': year,
                    'source': metadata.get('source', 'Unknown'),
                    'doc_id': i + 1
                })
    
    return interest_findings

def generate_interest_rate_answer(interest_findings, query):
    """Generate structured answer about interest rates with footnotes"""
    if not interest_findings:
        return "No interest rate information found in the documents."
    
    # Sort by rate value for better organization
    try:
        interest_findings.sort(key=lambda x: float(x['rate_value']) if x['rate_value'] != "N/A" else 0)
    except:
        pass
    
    answer = "## 📊 Interest Rate Information for Alaska Gas Pipeline Financing\n\n"
    
    # Generate summary
    rates = [f['rate_value'] for f in interest_findings if f['rate_value'] != "N/A"]
    if rates:
        try:
            rate_values = [float(r) for r in rates]
            min_rate = min(rate_values)
            max_rate = max(rate_values)
            answer += f"**Interest rates mentioned range from {min_rate}% to {max_rate}%**\n\n"
        except:
            pass
    
    # Add detailed findings with footnotes
    answer += "### 📋 Detailed Interest Rate References:\n\n"
    
    for i, finding in enumerate(interest_findings[:10], 1):
        answer += f"**{finding['rate']}** [{i}]\n"
        answer += f"   - **Year:** {finding['year']}\n"
        answer += f"   - **Context:** {finding['context'][:200]}...\n\n"
    
    # Add footnotes section
    answer += "---\n### 📚 Sources:\n\n"
    for i, finding in enumerate(interest_findings[:10], 1):
        answer += f"[{i}] {finding['filename']} ({finding['year']})\n"
    
    return answer

def extract_general_financial_info(documents, query):
    """Extract general financial information based on query"""
    financial_findings = []
    
    # Identify query type
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['cost', 'price', 'billion', 'million', 'dollar', '$']):
        # Cost-related patterns
        patterns = [
            r'\$[\d,]+(?:\.\d{1,2})?\s*(?:billion|million|thousand|B|M|K)',
            r'[\d,]+(?:\.\d{1,2})?\s*billion\s*(?:dollars?|USD|\$)',
            r'cost.*?\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:billion|million|thousand))?'
        ]
    elif any(word in query_lower for word in ['debt', 'financing', 'loan', 'borrow']):
        # Debt/financing patterns
        patterns = [
            r'debt\s*(?:of|is|was)?\s*\$?[\d,]+(?:\.\d{1,2})?\s*(?:billion|million)?',
            r'financing\s*(?:of|is|was)?\s*\$?[\d,]+(?:\.\d{1,2})?\s*(?:billion|million)?',
            r'loan\s*(?:of|is|was)?\s*\$?[\d,]+(?:\.\d{1,2})?\s*(?:billion|million)?'
        ]
    else:
        # General financial patterns
        patterns = [
            r'\$[\d,]+(?:\.\d{1,2})?\s*(?:billion|million|thousand)',
            r'[\d,]+(?:\.\d{1,2})?\s*(?:billion|million)\s*(?:dollars?|USD)'
        ]
    
    for i, doc in enumerate(documents):
        content = doc.page_content
        metadata = doc.metadata
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                finding_text = match.group()
                start_pos = max(0, match.start() - 250)
                end_pos = min(len(content), match.end() + 250)
                context = content[start_pos:end_pos].strip()
                
                financial_findings.append({
                    'finding': finding_text,
                    'context': context,
                    'filename': metadata.get('filename', 'Unknown'),
                    'source': metadata.get('source', 'Unknown'),
                    'doc_id': i + 1
                })
    
    return financial_findings

def generate_financial_answer(financial_findings, query):
    """Generate structured financial answer with citations"""
    if not financial_findings:
        return "No relevant financial information found in the documents."
    
    answer = f"## 💰 Financial Information Related to: *{query}*\n\n"
    
    # Group findings by document to avoid duplicate citations
    doc_groups = {}
    for finding in financial_findings:
        filename = finding['filename']
        if filename not in doc_groups:
            doc_groups[filename] = []
        doc_groups[filename].append(finding)
    
    answer += "### 📋 Key Findings:\n\n"
    
    citation_num = 1
    for filename, findings in doc_groups.items():
        answer += f"**From {filename}:** [{citation_num}]\n"
        
        for finding in findings[:3]:  # Limit to 3 findings per document
            answer += f"   • {finding['finding']}\n"
            answer += f"     *Context:* {finding['context'][:150]}...\n\n"
        
        citation_num += 1
    
    # Add footnotes
    answer += "---\n### 📚 Sources:\n\n"
    citation_num = 1
    for filename in doc_groups.keys():
        answer += f"[{citation_num}] {filename}\n"
        citation_num += 1
    
    return answer

def search_documents(query, documents, embeddings, top_k=15):
    """Enhanced search with better keyword matching"""
    if not documents:
        return []
    
    query_words = query.lower().split()
    relevant_docs = []
    
    for doc in documents:
        content_lower = doc.page_content.lower()
        
        # Enhanced scoring system
        interest_keywords = ['interest', 'rate', 'financing', 'debt', 'loan', 'borrow', 'cost of capital']
        financial_keywords = ['cost', 'price', 'billion', 'million', 'dollar', '$', 'budget', 'expense']
        pipeline_keywords = ['pipeline', 'gas', 'alaska', 'transport', 'lng']
        
        interest_score = sum(2 for word in interest_keywords if word in content_lower)
        financial_score = sum(1 for word in financial_keywords if word in content_lower)
        pipeline_score = sum(1 for word in pipeline_keywords if word in content_lower)
        query_score = sum(2 for word in query_words if word in content_lower)
        
        total_score = interest_score + financial_score + pipeline_score + query_score
        
        if total_score > 0:
            relevant_docs.append((doc, total_score))
    
    # Sort by relevance score
    relevant_docs.sort(key=lambda x: x[1], reverse=True)
    
    return