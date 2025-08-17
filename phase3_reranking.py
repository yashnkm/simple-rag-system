# Phase 3: Cross-Encoder Re-ranking Implementation  
# Add this code after installing: pip install sentence-transformers

def __init__(self):
    # Add these lines to your existing __init__ method
    self.reranker = None  # Cross-encoder for re-ranking

def load_reranker(self):
    """Load cross-encoder model for re-ranking"""
    try:
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Cross-encoder re-ranker loaded")
    except ImportError:
        print("⚠️ sentence-transformers not installed. Skipping re-ranking.")
        self.reranker = None
    except Exception as e:
        print(f"⚠️ Failed to load re-ranker: {e}")
        self.reranker = None

def rerank_results(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
    """Re-rank search results using cross-encoder"""
    if not self.reranker or not results:
        return results[:top_k]
    
    try:
        # Prepare query-document pairs for re-ranking
        query_doc_pairs = [(query, result["text"]) for result in results]
        
        # Get relevance scores from cross-encoder
        rerank_scores = self.reranker.predict(query_doc_pairs)
        
        # Add re-ranking scores to results
        for i, result in enumerate(results):
            result["rerank_score"] = float(rerank_scores[i])
            # Combine with existing scores
            existing_score = result.get("final_hybrid_score", result.get("final_score", 0.0))
            result["final_reranked_score"] = (existing_score * 0.3) + (result["rerank_score"] * 0.7)
        
        # Sort by re-ranked scores
        results.sort(key=lambda x: x["final_reranked_score"], reverse=True)
        return results[:top_k]
        
    except Exception as e:
        print(f"Re-ranking failed: {e}")
        return results[:top_k]

def search_with_reranking(self, query: str, k: int = 5) -> List[Dict[str, str]]:
    """Search with re-ranking for maximum accuracy"""
    # Get more candidates for re-ranking
    candidate_k = min(k * 3, 15)  # Get 3x more candidates
    
    # Use hybrid search if available
    if hasattr(self, 'bm25_index') and self.bm25_index:
        candidates = self.search_hybrid(query, k=candidate_k)
    else:
        candidates = self.search(query, k=candidate_k)
    
    # Re-rank candidates
    if self.reranker:
        final_results = self.rerank_results(query, candidates, top_k=k)
    else:
        final_results = candidates[:k]
    
    return final_results

# Update your generate_answer method to use re-ranking
def generate_answer(self, query: str) -> str:
    """Generate answer using RAG with re-ranking"""
    if not self.index:
        return "Please upload a document first."
    
    # Load re-ranker if not loaded
    if not hasattr(self, 'reranker'):
        self.load_reranker()
    
    # Use search with re-ranking
    relevant_results = self.search_with_reranking(query)
    
    # Rest of your existing generate_answer code...