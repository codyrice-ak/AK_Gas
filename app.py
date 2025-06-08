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

def enhanced_financial_search(query, embeddings_data, tfidf_data, top_k=20):
    """Enhanced search specifically optimized for financial queries"""
    
    if embeddings_data is None or tfidf_data is None:
        return []
    
    try:
        # Enhanced query preprocessing for financial terms
        financial_synonyms = {
            'interest rate': ['interest rate', 'financing rate', 'borrowing rate', 'cost of capital', 'debt rate'],
            'cost': ['cost', 'price', 'expense', 'expenditure', 'budget', 'investment'],
            'debt': ['debt', 'financing', 'loan', 'borrowing', 'credit'],
            'billion': ['billion', 'B', '000,000,000'],
            'million': ['million', 'M', '000,000']
        }
        
        # Expand query with financial synonyms
        expanded_query = query
        for term, synonyms in financial_synonyms.items():
            if term in query.lower():
                expanded_query += " " + " ".join(synonyms)
        
        # Use TF-IDF to encode the expanded query
        query_vector = tfidf_data['vectorizer'].transform([expanded_query])
        
        # Find most similar documents
        tfidf_similarities = cosine_similarity(query_vector, tfidf_data['tfidf_matrix']).flatten()
        
        # Enhanced scoring for financial content
        enhanced_scores = []
        for idx, base_score in enumerate(tfidf_similarities):
            content = embeddings_data['contents'][idx].lower()
            
            # Boost scores for financial indicators
            financial_boost = 0
            if any(term in content for term in ['interest', 'rate', '%', 'percent']):
                financial_boost += 0.2
            if any(term in content for term in ['billion', 'million', '$', 'cost']):
                financial_boost += 0.1
            if any(term in content for term in ['debt', 'financing', 'loan']):
                financial_boost += 0.1
            if any(term in content for term in ['pipeline', 'gas', 'alaska']):
                financial_boost += 0.05
                
            enhanced_scores.append(base_score + financial_boost)
        
        # Get top documents
        top_indices = np.argsort(enhanced_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if enhanced_scores[idx] > 0.01:
                results.append({
                    'content': embeddings_data['contents'][idx],
                    'metadata': embeddings_data['metadata'][idx],
                    'similarity': enhanced_scores[idx]
                })
        
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def extract_year_from_context(context):
    """Enhanced year extraction with multiple patterns"""
    year_patterns = [
        r'\b(19\d{2}|20\d{2})\b',  # Standard 4-digit years
        r'\bin\s+(19\d{2}|20\d{2})\b',  # "in 1977", "in 2019"
        r'\b(19\d{2}|20\d{2})\s+(?:estimate|study|report|analysis)\b'
    ]
    
    for pattern in year_patterns:
        matches = re.findall(pattern, context, re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    year = match[0] if len(match[0]) == 4 else match[1]
                else:
                    year = match
                if len(year) == 4:
                    return year
    
    return "Year not specified"

def extract_interest_rates_advanced(results):
    """Advanced interest rate extraction with comprehensive patterns"""
    interest_findings = []
    
    # Comprehensive interest rate patterns
    interest_patterns = [
        r'interest\s+rate[s]?\s*(?:of|at|is|was|are|were)?\s*([\d.]+)\s*(?:%|percent)',
        r'([\d.]+)\s*(?:%|percent)\s*interest\s*rate',
        r'financing\s*(?:at|with)?\s*([\d.]+)\s*(?:%|percent)',
        r'debt\s*(?:at|with|rate\s*of)?\s*([\d.]+)\s*(?:%|percent)',
        r'borrowing\s*(?:at|with|rate\s*of)?\s*([\d.]+)\s*(?:%|percent)',
        r'cost\s+of\s+capital\s*(?:of|at|is)?\s*([\d.]+)\s*(?:%|percent)',
        r'discount\s+rate\s*(?:of|at|is)?\s*([\d.]+)\s*(?:%|percent)',
        r'([\d.]+)\s*(?:%|percent)\s*(?:annual|yearly)?\s*(?:interest|financing|debt)',
        r'rate\s*(?:of|at)?\s*([\d.]+)\s*(?:%|percent)'
    ]
    
    for result in results:
        content = result['content']
        metadata = result['metadata']
        
        for pattern in interest_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                rate_text = match.group(0)
                rate_value = match.group(1) if len(match.groups()) > 0 else "N/A"
                
                # Get extended context
                start_pos = max(0, match.start() - 300)
                end_pos = min(len(content), match.end() + 300)
                context = content[start_pos:end_pos].strip()
                
                # Extract year from context
                year = extract_year_from_context(context)
                
                interest_findings.append({
                    'rate_text': rate_text,
                    'rate_value': float(rate_value) if rate_value != "N/A" and rate_value.replace('.', '').isdigit() else None,
                    'context': context,
                    'filename': metadata.get('filename', 'Unknown'),
                    'year': year,
                    'similarity': result['similarity']
                })
    
    return interest_findings

def generate_interest_rate_synthesis(interest_findings, query):
    """Generate sophisticated synthesized answer for interest rates"""
    if not interest_findings:
        return "No interest rate information found in the documents."
    
    # Filter valid rates and sort by value
    valid_rates = [f for f in interest_findings if f['rate_value'] is not None]
    valid_rates.sort(key=lambda x: x['rate_value'])
    
    answer = "## 📊 Interest Rate Analysis for Alaska Gas Pipeline Debt Financing\n\n"
    
    # Calculate statistics
    if valid_rates:
        rate_values = [f['rate_value'] for f in valid_rates]
        min_rate = min(rate_values)
        max_rate = max(rate_values)
        avg_rate = sum(rate_values) / len(rate_values)
        
        answer += f"**Analysis Summary:**\n"
        answer += f"- **Range:** {min_rate:.1f}% to {max_rate:.1f}%\n"
        answer += f"- **Average:** {avg_rate:.1f}%\n"
        answer += f"- **Number of estimates:** {len(valid_rates)}\n\n"
    
    # Group by time periods
    recent_rates = [f for f in valid_rates if f['year'] != "Year not specified" and int(f['year']) >= 1975]
    older_rates = [f for f in valid_rates if f['year'] != "Year not specified" and int(f['year']) < 1975]
    
    answer += "### 📋 Detailed Interest Rate Estimates:\n\n"
    
    # Show detailed findings with citations
    for i, finding in enumerate(valid_rates[:10], 1):
        answer += f"**{i}. {finding['rate_value']:.1f}%** [{i}]\n"
        answer += f"   - **Source:** {finding['filename']}\n"
        answer += f"   - **Year:** {finding['year']}\n"
        answer += f"   - **Context:** {finding['context'][:200]}...\n\n"
    
    # Add footnotes section
    answer += "---\n### 📚 Sources:\n\n"
    for i, finding in enumerate(valid_rates[:10], 1):
        answer += f"[{i}] {finding['filename']} ({finding['year']})\n"
    
    return answer

def extract_cost_estimates_advanced(results):
    """Advanced cost extraction with comprehensive patterns"""
    cost_findings = []
    
    # Enhanced cost patterns
    cost_patterns = [
        r'\$\s*([\d,]+(?:\.\d{1,2})?)\s*(billion|million|thousand|B|M|K)',
        r'([\d,]+(?:\.\d{1,2})?)\s*(billion|million)\s*(?:dollars?|USD|\$)',
        r'cost[s]?\s*(?:of|is|was|estimated|projected)?\s*(?:at|to\s+be)?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)\s*(billion|million|thousand)?',
        r'estimate[d]?\s*(?:at|to\s+be)?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)\s*(billion|million|thousand)',
        r'budget[s]?\s*(?:of|is|was)?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)\s*(billion|million|thousand)',
        r'investment[s]?\s*(?:of|is|was)?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)\s*(billion|million|thousand)'
    ]
    
    for result in results:
        content = result['content']
        metadata = result['metadata']
        
        for pattern in cost_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                cost_text = match.group(0)
                
                # Extract numerical value and unit
                groups = match.groups()
                if len(groups) >= 2:
                    value_str = groups[0].replace(',', '')
                    unit = groups[1].lower() if groups[1] else ''
                    
                    try:
                        value = float(value_str)
                        # Normalize to billions
                        if unit in ['million', 'm']:
                            normalized_value = value / 1000
                        elif unit in ['thousand', 'k']:
                            normalized_value = value / 1000000
                        else:  # billion or B
                            normalized_value = value
                    except:
                        normalized_value = None
                else:
                    normalized_value = None
                
                # Get context
                start_pos = max(0, match.start() - 250)
                end_pos = min(len(content), match.end() + 250)
                context = content[start_pos:end_pos].strip()
                
                year = extract_year_from_context(context)
                
                cost_findings.append({
                    'cost_text': cost_text,
                    'normalized_value': normalized_value,
                    'context': context,
                    'filename': metadata.get('filename', 'Unknown'),
                    'year': year,
                    'similarity': result['similarity']
                })
    
    return cost_findings

def main():
    st.title("🛢️ Alaska Gas Pipeline Document Query System")
    
    # Load data
    embeddings_data = load_embeddings_data()
    tfidf_data = load_tfidf_data()
    
    if embeddings_data and tfidf_data:
        st.success(f"✅ Loaded {len(embeddings_data['contents']):,} document embeddings for semantic search!")
        st.info("🔍 Using enhanced financial semantic search with specialized analysis")
    else:
        st.error("❌ Could not load embeddings data")
        return
    
    # Query interface
    user_query = st.chat_input("Ask your question about Alaska Gas Pipeline documents...")
    
    if user_query:
        st.write(f"**Your question:** {user_query}")
        
        with st.spinner("Performing enhanced financial analysis..."):
            results = enhanced_financial_search(user_query, embeddings_data, tfidf_data)
        
        if results:
            # Determine query type and provide specialized analysis
            if any(word in user_query.lower() for word in ['interest', 'rate', 'financing', 'debt']):
                # Interest rate specific analysis
                interest_findings = extract_interest_rates_advanced(results)
                answer = generate_interest_rate_synthesis(interest_findings, user_query)
                st.markdown(answer)
                
            elif any(word in user_query.lower() for word in ['cost', 'price', 'billion', 'million', 'dollar', '$', 'budget']):
                # Cost analysis
                cost_findings = extract_cost_estimates_advanced(results)
                
                if cost_findings:
                    valid_costs = [f for f in cost_findings if f['normalized_value'] is not None]
                    
                    st.markdown("## 💰 Cost Analysis for Alaska Gas Pipeline\n")
                    
                    if valid_costs:
                        cost_values = [f['normalized_value'] for f in valid_costs]
                        min_cost = min(cost_values)
                        max_cost = max(cost_values)
                        avg_cost = sum(cost_values) / len(cost_values)
                        
                        st.markdown(f"**Cost Range:** ${min_cost:.1f} - ${max_cost:.1f} billion USD\n")
                        st.markdown(f"**Average Estimate:** ${avg_cost:.1f} billion USD\n")
                    
                    st.markdown("### 📋 Detailed Cost Estimates:\n")
                    
                    for i, finding in enumerate(cost_findings[:10], 1):
                        st.markdown(f"**{i}. {finding['cost_text']}** [{i}]")
                        st.markdown(f"   - **Source:** {finding['filename']}")
                        st.markdown(f"   - **Year:** {finding['year']}")
                        st.markdown(f"   - **Context:** {finding['context'][:200]}...\n")
                    
                    # Add footnotes
                    st.markdown("---\n### 📚 Sources:")
                    for i, finding in enumerate(cost_findings[:10], 1):
                        st.markdown(f"[{i}] {finding['filename']} ({finding['year']})")
                else:
                    st.warning("No cost information found in the documents.")
            
            else:
                # General analysis with enhanced formatting
                st.write("**📋 Relevant Information Found:**")
                for i, result in enumerate(results[:5], 1):
                    with st.expander(f"Result {i}: {result['metadata'].get('filename', 'Unknown')} (Relevance: {result['similarity']:.3f})"):
                        st.write(result['content'])
                        st.caption(f"Source: {result['metadata'].get('filename', 'Unknown')}")
        else:
            st.warning("No relevant documents found. Try rephrasing your question with different financial terms.")

if __name__ == "__main__":
    main()