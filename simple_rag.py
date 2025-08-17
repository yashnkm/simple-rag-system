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
    
    def chunk_text(self, text: str, chunk_size: int = 2500) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            result = genai.embed_content(
                model=self.embeddings_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 768  # Return zero vector on error
    
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
            
            # Create status message
            total_docs = len(self.documents)
            total_chunks = len(self.texts)
            return f"✅ Document '{file.name}' processed successfully!\n📚 Total documents: {total_docs}\n📄 Total chunks: {total_chunks}\n🆕 New chunks: {len(new_texts)}"
                
        except Exception as e:
            return f"Error processing document: {str(e)}"
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """Search for relevant chunks with document source"""
        if not self.index or not self.texts:
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # Search in FAISS
        query_array = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_array, k)
        
        # Get relevant texts with document info
        relevant_results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                doc_id = self.text_to_doc[idx]
                doc_name = self.documents[doc_id]["name"]
                relevant_results.append({
                    "text": self.texts[idx],
                    "document": doc_name,
                    "doc_id": doc_id
                })
        
        return relevant_results
    
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
        """Generate answer using RAG with conversation memory"""
        if not self.index:
            return "Please upload a document first."
        
        # Get relevant context
        relevant_results = self.search(query)
        if not relevant_results:
            return "No relevant information found in the documents."
        
        # Build document context with sources
        document_parts = []
        sources_used = set()
        for result in relevant_results:
            document_parts.append(f"[From {result['document']}]: {result['text']}")
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

Answer:"""
        
        try:
            response = self.llm_model.generate_content(prompt)
            answer = response.text
            
            # Add source information to answer
            if len(sources_used) > 0:
                answer += f"\n\n📚 Sources: {sources_list}"
            
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