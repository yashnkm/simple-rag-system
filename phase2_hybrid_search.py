# Phase 2: Hybrid Search Implementation
# Add this code to your SimpleRAG class after installing: pip install rank-bm25

def __init__(self):
    # Add these lines to your existing __init__ method
    self.bm25_index = None  # BM25 keyword search index
    self.tokenized_texts = []  # Tokenized texts for BM25

def build_bm25_index(self):
    """Build BM25 index for keyword search"""
    if not self.texts:
        self.bm25_index = None
        return
    
    try:
        from rank_bm25 import BM25Okapi
        
        # Tokenize texts for BM25
        self.tokenized_texts = [text.lower().split() for text in self.texts]
        self.bm25_index = BM25Okapi(self.tokenized_texts)
        print(f"✅ BM25 index built with {len(self.texts)} documents")
    except ImportError:
        print("⚠️ rank-bm25 not installed. Using semantic search only.")
        self.bm25_index = None

def search_hybrid(self, query: str, k: int = 5) -> List[Dict[str, str]]:
    """Hybrid search combining BM25 + semantic search"""
    if not self.index or not self.texts:
        return []
    
    # Preprocess query
    processed_query = self.preprocess_query(query)
    query_tokens = processed_query.lower().split()
    
    # Get semantic search results
    semantic_results = self.search(processed_query, k=k*2)  # Get more candidates
    
    # Get BM25 keyword search results
    bm25_results = []
    if self.bm25_index:
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top BM25 results
        bm25_top_indices = sorted(range(len(bm25_scores)), 
                                 key=lambda i: bm25_scores[i], reverse=True)[:k*2]
        
        for idx in bm25_top_indices:
            if idx < len(self.texts) and bm25_scores[idx] > 0:
                doc_id = self.text_to_doc[idx]
                doc_name = self.documents[doc_id]["name"]
                
                bm25_results.append({
                    "text": self.texts[idx],
                    "document": doc_name,
                    "doc_id": doc_id,
                    "bm25_score": float(bm25_scores[idx]),
                    "index": idx
                })
    
    # Combine and deduplicate results
    combined_results = {}
    
    # Add semantic results
    for result in semantic_results:
        text_hash = hash(result["text"])
        combined_results[text_hash] = {
            **result,
            "semantic_score": result.get("final_score", 0.0),
            "bm25_score": 0.0
        }
    
    # Add/update with BM25 results
    for result in bm25_results:
        text_hash = hash(result["text"])
        if text_hash in combined_results:
            combined_results[text_hash]["bm25_score"] = result["bm25_score"]
        else:
            combined_results[text_hash] = {
                **result,
                "semantic_score": 0.0,
                "bm25_score": result["bm25_score"]
            }
    
    # Calculate final hybrid scores
    final_results = []
    for result in combined_results.values():
        # Normalize scores (simple min-max normalization)
        semantic_norm = result["semantic_score"]
        bm25_norm = result["bm25_score"] / max(1.0, max([r["bm25_score"] for r in combined_results.values()]))
        
        # Weighted combination: 60% semantic, 40% keyword
        final_score = (semantic_norm * 0.6) + (bm25_norm * 0.4)
        
        result["final_hybrid_score"] = final_score
        final_results.append(result)
    
    # Sort by hybrid score and return top k
    final_results.sort(key=lambda x: x["final_hybrid_score"], reverse=True)
    return final_results[:k]

# Update your process_document method to rebuild BM25 index
def process_document(self, file) -> str:
    # Your existing code...
    # Add this line at the end, before the return statement:
    self.build_bm25_index()  # Rebuild BM25 index after adding document
    # return your existing return statement

# Update your generate_answer method
def generate_answer(self, query: str) -> str:
    """Generate answer using hybrid RAG"""
    if not self.index:
        return "Please upload a document first."
    
    # Use hybrid search if BM25 is available, otherwise use enhanced search
    if self.bm25_index:
        relevant_results = self.search_hybrid(query)
    else:
        relevant_results = self.search(query)
    
    # Rest of your existing generate_answer code...