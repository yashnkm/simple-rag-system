import os
import io
import google.generativeai as genai
import PyPDF2
import numpy as np
import faiss
import gradio as gr
from dotenv import load_dotenv
from typing import List, Tuple, Dict
from datetime import datetime
import re

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('rag_google_office_use'))

class SimpleRAG:
    def __init__(self):
        self.embeddings_model = "models/text-embedding-004"
        self.llm_model = genai.GenerativeModel('gemini-2.5-flash')
        self.texts = []  # List of all text chunks
        self.embeddings = []  # List of all embeddings
        self.index = None  # Combined FAISS index
        self.conversation_history = []  # Keep last 7 conversations
        self.documents = {}  # Document metadata: {doc_id: {name, upload_time, chunk_count, chunk_indices}}
        self.text_to_doc = []  # Maps text index to document ID
        self.bm25_index = None  # BM25 keyword search index
        self.tokenized_texts = []  # Tokenized texts for BM25
        self.reranker = None  # Cross-encoder for re-ranking
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_txt(self, txt_file) -> str:
        """Extract text from TXT file"""
        try:
            return txt_file.read().decode('utf-8')
        except Exception as e:
            return f"Error reading TXT: {str(e)}"
    
    def chunk_text_smart(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Smart text chunking with sentence awareness and overlap"""
        # Split into sentences using regex
        sentence_endings = r'[.!?]+[\s\n]+'
        sentences = re.split(sentence_endings, text.strip())
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence.split())
            
            # If single sentence is too long, split it
            if sentence_length > chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Split long sentence into smaller parts
                words = sentence.split()
                for i in range(0, len(words), chunk_size - 50):  # Leave some buffer
                    chunk_part = ' '.join(words[i:i + chunk_size - 50])
                    if chunk_part.strip():
                        chunks.append(chunk_part.strip())
                
                current_chunk = ""
                current_length = 0
                continue
            
            # Check if adding this sentence exceeds chunk size
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap by keeping last few sentences
                if overlap > 0:
                    overlap_sentences = current_chunk.split('. ')[-2:]  # Last 2 sentences
                    overlap_text = '. '.join(overlap_sentences)
                    if len(overlap_text.split()) <= overlap:
                        current_chunk = overlap_text + '. ' + sentence
                        current_length = len(current_chunk.split())
                    else:
                        current_chunk = sentence
                        current_length = sentence_length
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += '. ' + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.split()) >= 10]
        
        return chunks
    
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Main chunking method - now uses smart chunking"""
        return self.chunk_text_smart(text, chunk_size=chunk_size, overlap=200)
    
    def preprocess_query(self, query: str) -> str:
        """Enhance query for better retrieval"""
        # Remove extra whitespace and normalize
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Expand common abbreviations and terms
        expansions = {
            r'\bwhat\'s\b': 'what is',
            r'\bhow\'s\b': 'how is', 
            r'\bcan\'t\b': 'cannot',
            r'\bwon\'t\b': 'will not',
            r'\bdon\'t\b': 'do not',
            r'\bML\b': 'machine learning',
            r'\bAI\b': 'artificial intelligence',
            r'\bNLP\b': 'natural language processing',
            r'\bAPI\b': 'application programming interface'
        }
        
        for pattern, replacement in expansions.items():
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        
        return query
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            # Preprocess text for better embeddings
            processed_text = re.sub(r'\s+', ' ', text.strip())
            
            result = genai.embed_content(
                model=self.embeddings_model,
                content=processed_text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 768  # Return zero vector on error
    
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
        except Exception as e:
            print(f"⚠️ BM25 index creation failed: {e}")
            self.bm25_index = None
    
    def load_reranker(self):
        """Load cross-encoder model for re-ranking"""
        if self.reranker is not None:
            return  # Already loaded
            
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
    
    def process_document(self, file) -> str:
        """Process uploaded document and add to collection"""
        try:
            # Extract text based on file type
            if file.name.endswith('.pdf'):
                text = self.extract_text_from_pdf(file)
            elif file.name.endswith('.txt'):
                text = self.extract_text_from_txt(file)
            else:
                return "Unsupported file type. Please upload PDF or TXT file."
            
            # Chunk the text
            chunks = self.chunk_text(text)
            if not chunks:
                return "No text found in document."
            
            # Generate document ID
            doc_id = f"doc_{len(self.documents) + 1}_{datetime.now().strftime('%H%M%S')}"
            
            # Track starting index for this document
            start_idx = len(self.texts)
            
            # Generate embeddings for each chunk
            new_embeddings = []
            new_texts = []
            for chunk in chunks:
                embedding = self.get_embedding(chunk)
                if embedding:
                    new_texts.append(chunk)
                    new_embeddings.append(embedding)
                    self.text_to_doc.append(doc_id)  # Map chunk to document
            
            if not new_embeddings:
                return "Failed to create embeddings."
            
            # Add to existing data
            self.texts.extend(new_texts)
            self.embeddings.extend(new_embeddings)
            
            # Store document metadata
            self.documents[doc_id] = {
                "name": file.name,
                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "chunk_count": len(new_texts),
                "chunk_indices": list(range(start_idx, start_idx + len(new_texts)))
            }
            
            # Rebuild FAISS index with all embeddings
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
            self.index.add(embeddings_array)
            
            # Rebuild BM25 index
            self.build_bm25_index()
            
            # Create status message
            total_docs = len(self.documents)
            total_chunks = len(self.texts)
            bm25_status = "✅ BM25" if self.bm25_index else "⚠️ No BM25"
            return f"✅ Document '{file.name}' processed successfully!\n📚 Total documents: {total_docs}\n📄 Total chunks: {total_chunks}\n🆕 New chunks: {len(new_texts)}\n🔍 Search: {bm25_status} + Semantic"
                
        except Exception as e:
            return f"Error processing document: {str(e)}"
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """Enhanced search with query preprocessing and adaptive retrieval"""
        if not self.index or not self.texts:
            return []
        
        # Preprocess query for better matching
        processed_query = self.preprocess_query(query)
        
        # Get query embedding
        query_embedding = self.get_embedding(processed_query)
        if not query_embedding:
            return []
        
        # Adaptive k based on query complexity
        query_words = len(processed_query.split())
        if query_words <= 3:
            k = min(k, 3)  # Simple queries need fewer chunks
        elif query_words > 10:
            k = min(k + 2, 8)  # Complex queries might need more chunks
        
        # Search in FAISS with more candidates for potential re-ranking
        search_k = min(k * 2, len(self.texts))  # Get more candidates
        query_array = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_array, search_k)
        
        # Get relevant texts with document info and scores
        relevant_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                doc_id = self.text_to_doc[idx]
                doc_name = self.documents[doc_id]["name"]
                
                # Simple relevance scoring based on distance
                similarity_score = 1.0 / (1.0 + distances[0][i])
                
                relevant_results.append({
                    "text": self.texts[idx],
                    "document": doc_name,
                    "doc_id": doc_id,
                    "score": similarity_score,
                    "distance": float(distances[0][i])
                })
        
        # Basic re-ranking: prefer results with query terms
        def calculate_keyword_score(text: str, query: str) -> float:
            text_lower = text.lower()
            query_words = query.lower().split()
            
            score = 0.0
            for word in query_words:
                if len(word) > 2:  # Skip very short words
                    if word in text_lower:
                        score += 1.0
                        # Bonus for exact phrase matches
                        if word in text_lower.split():
                            score += 0.5
            
            return score / len(query_words) if query_words else 0.0
        
        # Add keyword scores and re-rank
        for result in relevant_results:
            keyword_score = calculate_keyword_score(result["text"], processed_query)
            # Combine semantic and keyword scores
            result["final_score"] = (result["score"] * 0.7) + (keyword_score * 0.3)
        
        # Sort by final score and return top k
        relevant_results.sort(key=lambda x: x["final_score"], reverse=True)
        return relevant_results[:k]
    
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
            try:
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
            except Exception as e:
                print(f"BM25 search failed: {e}")
        
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
        max_bm25 = max([r["bm25_score"] for r in combined_results.values()]) if combined_results else 1.0
        
        for result in combined_results.values():
            # Normalize scores
            semantic_norm = result["semantic_score"]
            bm25_norm = result["bm25_score"] / max(1.0, max_bm25)
            
            # Weighted combination: 60% semantic, 40% keyword
            final_score = (semantic_norm * 0.6) + (bm25_norm * 0.4)
            
            result["final_hybrid_score"] = final_score
            final_results.append(result)
        
        # Sort by hybrid score and return top k
        final_results.sort(key=lambda x: x["final_hybrid_score"], reverse=True)
        return final_results[:k]
    
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
        if self.bm25_index:
            candidates = self.search_hybrid(query, k=candidate_k)
        else:
            candidates = self.search(query, k=candidate_k)
        
        # Re-rank candidates if re-ranker is available
        if self.reranker:
            final_results = self.rerank_results(query, candidates, top_k=k)
        else:
            final_results = candidates[:k]
        
        return final_results
    
    def get_document_list(self) -> str:
        """Get formatted list of uploaded documents"""
        if not self.documents:
            return "No documents uploaded yet."
        
        doc_list = []
        for doc_id, metadata in self.documents.items():
            doc_list.append(f"📄 {metadata['name']} ({metadata['chunk_count']} chunks, {metadata['upload_time']})")
        
        return "\n".join(doc_list)
    
    def get_document_choices(self) -> List[str]:
        """Get list of document names for dropdown"""
        if not self.documents:
            return ["No documents to remove"]
        
        return [f"{metadata['name']}" for doc_id, metadata in self.documents.items()]
    
    def remove_document(self, doc_name: str) -> str:
        """Remove a specific document and rebuild index"""
        if not self.documents:
            return "No documents to remove."
        
        # Find document ID by name
        doc_id_to_remove = None
        for doc_id, metadata in self.documents.items():
            if metadata['name'] == doc_name:
                doc_id_to_remove = doc_id
                break
        
        if not doc_id_to_remove:
            return f"Document '{doc_name}' not found."
        
        try:
            # Get indices to remove
            chunk_indices = self.documents[doc_id_to_remove]["chunk_indices"]
            
            # Remove from texts, embeddings, and text_to_doc (in reverse order to maintain indices)
            for idx in sorted(chunk_indices, reverse=True):
                if idx < len(self.texts):
                    del self.texts[idx]
                    del self.embeddings[idx]
                    del self.text_to_doc[idx]
            
            # Update chunk indices for remaining documents
            for doc_id, metadata in self.documents.items():
                if doc_id != doc_id_to_remove:
                    new_indices = []
                    for idx in metadata["chunk_indices"]:
                        # Count how many indices were removed before this one
                        removed_before = sum(1 for removed_idx in chunk_indices if removed_idx < idx)
                        new_indices.append(idx - removed_before)
                    metadata["chunk_indices"] = new_indices
            
            # Remove document metadata
            del self.documents[doc_id_to_remove]
            
            # Rebuild FAISS index
            if self.embeddings:
                embeddings_array = np.array(self.embeddings, dtype=np.float32)
                self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
                self.index.add(embeddings_array)
            else:
                self.index = None
            
            remaining_docs = len(self.documents)
            total_chunks = len(self.texts)
            return f"✅ Document '{doc_name}' removed successfully!\n📚 Remaining documents: {remaining_docs}\n📄 Total chunks: {total_chunks}"
            
        except Exception as e:
            return f"Error removing document: {str(e)}"
    
    def clear_all_documents(self) -> str:
        """Clear all documents and reset the system"""
        self.texts = []
        self.embeddings = []
        self.index = None
        self.documents = {}
        self.text_to_doc = []
        self.conversation_history = []  # Also clear conversation history
        
        return "✅ All documents and conversation history cleared!"
    
    def add_to_conversation_history(self, question: str, answer: str):
        """Add Q&A pair to conversation history, keep only last 7"""
        self.conversation_history.append({"question": question, "answer": answer})
        if len(self.conversation_history) > 7:
            self.conversation_history = self.conversation_history[-7:]
    
    def get_conversation_context(self) -> str:
        """Build conversation context from history"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for i, conv in enumerate(self.conversation_history[-3:], 1):  # Last 3 for context
            context_parts.append(f"Previous Q{i}: {conv['question']}")
            context_parts.append(f"Previous A{i}: {conv['answer']}")
        
        return "\n".join(context_parts)

    def generate_answer(self, query: str) -> str:
        """Generate answer using advanced RAG with hybrid search and re-ranking"""
        if not self.index:
            return "Please upload a document first."
        
        # Load re-ranker if not loaded and available
        if not self.reranker:
            self.load_reranker()
        
        # Use the best available search method
        if self.reranker:
            relevant_results = self.search_with_reranking(query)
            search_method = "🧠 Re-ranked"
        elif self.bm25_index:
            relevant_results = self.search_hybrid(query)
            search_method = "🔍 Hybrid"
        else:
            relevant_results = self.search(query)
            search_method = "🔎 Semantic"
        
        if not relevant_results:
            return "No relevant information found in the documents."
        
        # Build document context with sources and relevance info
        document_parts = []
        sources_used = set()
        for i, result in enumerate(relevant_results, 1):
            # Add relevance indicator
            score = result.get("final_reranked_score", result.get("final_hybrid_score", result.get("final_score", 0.0)))
            relevance = "🎯" if score > 0.7 else "📄" if score > 0.4 else "📃"
            
            document_parts.append(f"[{relevance} From {result['document']}]: {result['text']}")
            sources_used.add(result['document'])
        
        document_context = "\n\n".join(document_parts)
        sources_list = ", ".join(sources_used)
        
        # Build conversation context
        conversation_context = self.get_conversation_context()
        
        # Generate prompt with conversation history
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Your answers should be human-like, conversational, and easy to understand.
You can reference previous parts of our conversation when relevant.

{conversation_context}

Document Context (from multiple sources):
{document_context}

Current Question: {query}

Instructions:
- Answer based on the document context provided
- Reference previous conversation when relevant
- Be conversational and natural
- If the context doesn't have enough information, say so politely
- Keep the answer clear and concise
- When relevant, mention which document(s) you're referencing
- The 🎯 emoji indicates high-relevance content, prioritize that information

Answer:"""
        
        try:
            response = self.llm_model.generate_content(prompt)
            answer = response.text
            
            # Add source and method information to answer
            if len(sources_used) > 0:
                answer += f"\n\n📚 Sources: {sources_list}\n🔍 Search: {search_method}"
            
            # Add to conversation history
            self.add_to_conversation_history(query, answer)
            
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Initialize RAG system
rag = SimpleRAG()

# Gradio interface functions
def upload_document(file):
    """Handle document upload"""
    if file is None:
        return "Please upload a file.", "No documents uploaded yet.", gr.update(choices=["No documents to remove"])
    result = rag.process_document(file)
    doc_list = rag.get_document_list()
    doc_choices = rag.get_document_choices()
    return result, doc_list, gr.update(choices=doc_choices)

def remove_selected_document(doc_name):
    """Handle document removal"""
    if doc_name == "No documents to remove":
        return "No documents to remove.", "No documents uploaded yet.", gr.update(choices=["No documents to remove"])
    
    result = rag.remove_document(doc_name)
    doc_list = rag.get_document_list()
    doc_choices = rag.get_document_choices()
    return result, doc_list, gr.update(choices=doc_choices)

def clear_all_docs():
    """Handle clearing all documents"""
    result = rag.clear_all_documents()
    doc_list = rag.get_document_list()
    doc_choices = rag.get_document_choices()
    return result, doc_list, gr.update(choices=doc_choices)

def chat_function(message, history):
    """Handle chat messages with history"""
    if not message:
        return "Please enter a question."
    
    # Generate answer using RAG
    response = rag.generate_answer(message)
    return response

# Create Gradio interface with ChatInterface
with gr.Blocks(title="Simple RAG System with Memory") as demo:
    gr.Markdown("# 🤖 Simple RAG System with Conversation Memory")
    gr.Markdown("Upload a PDF or TXT document and have a conversation about it!")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Document Upload")
            file_input = gr.File(
                label="Upload Document (PDF or TXT)",
                file_types=[".pdf", ".txt"]
            )
            upload_btn = gr.Button("Process Document", variant="primary")
            upload_output = gr.Textbox(label="Upload Status", lines=3, interactive=False)
            
            gr.Markdown("### 📚 Uploaded Documents")
            doc_list_output = gr.Textbox(
                label="Document Collection", 
                lines=4, 
                interactive=False,
                value="No documents uploaded yet."
            )
            
            gr.Markdown("### 🗑️ Remove Documents")
            with gr.Row():
                doc_dropdown = gr.Dropdown(
                    label="Select Document to Remove",
                    choices=["No documents to remove"],
                    value="No documents to remove"
                )
            with gr.Row():
                remove_btn = gr.Button("Remove Selected", variant="secondary", scale=1)
                clear_all_btn = gr.Button("Clear All", variant="stop", scale=1)
            
            gr.Markdown("### 💡 Example Questions")
            gr.Markdown("""
            - What is the main topic of these documents?
            - Can you summarize the key points?
            - Tell me more about that
            - What are the important findings?
            - How does this relate to what we discussed earlier?
            - Compare information between the documents
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with Your Documents")
            chat_interface = gr.ChatInterface(
                fn=chat_function,
                examples=[
                    "What is this document about?",
                    "Can you explain that in simpler terms?",
                    "What did we just discuss?",
                ]
            )
    
    # Connect functions
    upload_btn.click(
        upload_document, 
        inputs=file_input, 
        outputs=[upload_output, doc_list_output, doc_dropdown]
    )
    
    remove_btn.click(
        remove_selected_document,
        inputs=doc_dropdown,
        outputs=[upload_output, doc_list_output, doc_dropdown]
    )
    
    clear_all_btn.click(
        clear_all_docs,
        inputs=None,
        outputs=[upload_output, doc_list_output, doc_dropdown]
    )

if __name__ == "__main__":
    print("Starting Simple RAG System...")
    print("Opening browser at http://localhost:7860")
    demo.launch(server_name="localhost", server_port=7860)