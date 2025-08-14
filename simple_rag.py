import os
import io
import google.generativeai as genai
import PyPDF2
import numpy as np
import faiss
import gradio as gr
from dotenv import load_dotenv
from typing import List, Tuple

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('rag_google_office_use'))

class SimpleRAG:
    def __init__(self):
        self.embeddings_model = "models/text-embedding-004"
        self.llm_model = genai.GenerativeModel('gemini-2.5-flash')
        self.texts = []
        self.embeddings = []
        self.index = None
        
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
        """Process uploaded document"""
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
            
            # Clear existing data
            self.texts = []
            self.embeddings = []
            
            # Generate embeddings for each chunk
            for chunk in chunks:
                embedding = self.get_embedding(chunk)
                if embedding:
                    self.texts.append(chunk)
                    self.embeddings.append(embedding)
            
            # Create FAISS index
            if self.embeddings:
                embeddings_array = np.array(self.embeddings, dtype=np.float32)
                self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
                self.index.add(embeddings_array)
                return f"✅ Document processed successfully! {len(self.texts)} chunks created."
            else:
                return "Failed to create embeddings."
                
        except Exception as e:
            return f"Error processing document: {str(e)}"
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant chunks"""
        if not self.index or not self.texts:
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # Search in FAISS
        query_array = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_array, k)
        
        # Get relevant texts
        relevant_texts = []
        for idx in indices[0]:
            if idx < len(self.texts):
                relevant_texts.append(self.texts[idx])
        
        return relevant_texts
    
    def generate_answer(self, query: str) -> str:
        """Generate answer using RAG"""
        if not self.index:
            return "Please upload a document first."
        
        # Get relevant context
        relevant_chunks = self.search(query)
        if not relevant_chunks:
            return "No relevant information found in the document."
        
        # Build context
        context = "\n\n".join(relevant_chunks)
        
        # Generate prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Your answers should be human-like, conversational, and easy to understand.

Context from document:
{context}

Question: {query}

Instructions:
- Answer based on the context provided
- Be conversational and natural
- If the context doesn't have enough information, say so politely
- Keep the answer clear and concise

Answer:"""
        
        try:
            response = self.llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Initialize RAG system
rag = SimpleRAG()

# Gradio interface
def upload_document(file):
    """Handle document upload"""
    if file is None:
        return "Please upload a file."
    return rag.process_document(file)

def ask_question(question):
    """Handle question answering"""
    if not question:
        return "Please enter a question."
    return rag.generate_answer(question)

# Create Gradio interface
with gr.Blocks(title="Simple RAG System") as demo:
    gr.Markdown("# 🤖 Simple RAG System")
    gr.Markdown("Upload a PDF or TXT document and ask questions about it!")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Upload Document (PDF or TXT)",
                file_types=[".pdf", ".txt"]
            )
            upload_btn = gr.Button("Process Document", variant="primary")
            upload_output = gr.Textbox(label="Status", lines=2)
            
        with gr.Column():
            question_input = gr.Textbox(
                label="Ask a Question",
                placeholder="What is this document about?",
                lines=3
            )
            ask_btn = gr.Button("Get Answer", variant="primary")
            answer_output = gr.Textbox(label="Answer", lines=10)
    
    # Connect functions
    upload_btn.click(upload_document, inputs=file_input, outputs=upload_output)
    ask_btn.click(ask_question, inputs=question_input, outputs=answer_output)
    
    gr.Examples(
        examples=[
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the important findings?",
        ],
        inputs=question_input
    )

if __name__ == "__main__":
    print("Starting Simple RAG System...")
    print("Opening browser at http://localhost:7860")
    demo.launch(server_name="localhost", server_port=7860)