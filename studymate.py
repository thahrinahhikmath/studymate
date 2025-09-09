import streamlit as st
import os
import time
import torch
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import tempfile

# StudyMate Classes - All in one file
class PDFProcessor:
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            st.error(f"Error extracting PDF: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
            
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

class VectorStore:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            st.warning(f"Using fallback embedding model due to: {e}")
            self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        
        self.index = None
        self.texts = []
        
    def add_texts(self, texts: List[str]):
        """Add texts to vector store"""
        if not texts:
            return
            
        try:
            self.texts.extend(texts)
            embeddings = self.embedding_model.encode(texts)
            
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
            
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
        except Exception as e:
            st.error(f"Error adding texts to vector store: {e}")
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar texts"""
        if self.index is None or len(self.texts) == 0:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            k = min(k, len(self.texts))
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.texts) and idx >= 0:
                    results.append({
                        'text': self.texts[idx],
                        'score': float(score),
                        'index': int(idx)
                    })
            
            return results
        except Exception as e:
            st.error(f"Error searching: {e}")
            return []

class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.exchanges = []
        self.max_history = max_history
    
    def add_exchange(self, question: str, answer: str, metadata: Dict = None):
        exchange = {
            'question': question,
            'answer': answer,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.exchanges.append(exchange)
        
        if len(self.exchanges) > self.max_history:
            self.exchanges = self.exchanges[-self.max_history:]
    
    def get_recent_context(self, n: int = 3) -> List[Dict]:
        return self.exchanges[-n:] if self.exchanges else []
    
    def get_relevant_context(self, question: str, n: int = 2) -> List[Dict]:
        return self.exchanges[-n:] if self.exchanges else []

class StudyMate:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.memory = ConversationMemory()
        
        # Initialize QA pipeline
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
        except Exception as e:
            st.warning(f"Using fallback QA model: {e}")
            self.qa_pipeline = pipeline("question-answering")
        
    def upload_and_process_pdf(self, pdf_path: str):
        """Process uploaded PDF"""
        try:
            # Extract text
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            # Chunk text
            chunks = self.pdf_processor.chunk_text(text)
            if not chunks:
                raise ValueError("Could not create text chunks from PDF")
            
            # Add to vector store
            self.vector_store.add_texts(chunks)
            
            return len(chunks)
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            raise e
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Answer question based on uploaded documents"""
        start_time = time.time()
        
        try:
            search_results = self.vector_store.search(question, k=3)
            
            if not search_results:
                return {
                    'answer': "No relevant information found. Please upload a PDF document first.",
                    'confidence': 0.0,
                    'sources': [],
                    'query_time': time.time() - start_time
                }
            
            context = " ".join([result['text'] for result in search_results])
            
            # Truncate context if too long
            context_words = context.split()
            if len(context_words) > 400:
                context = " ".join(context_words[:400])
            
            result = self.qa_pipeline(question=question, context=context)
            
            response = {
                'answer': result['answer'],
                'confidence': result['score'],
                'sources': search_results[:2],
                'query_time': time.time() - start_time
            }
            
            self.memory.add_exchange(question, result['answer'], {'confidence': result['score']})
            
            return response
            
        except Exception as e:
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'query_time': time.time() - start_time
            }

# Streamlit App
st.set_page_config(
    page_title="StudyMate - AI PDF Q&A", 
    page_icon="🎓", 
    layout="wide"
)

@st.cache_resource
def load_studymate():
    return StudyMate()

def main():
    st.title("🎓 StudyMate - AI-Powered PDF Q&A")
    st.markdown("Upload your PDF documents and ask questions!")
    
    # Initialize StudyMate
    if 'studymate' not in st.session_state:
        with st.spinner("Initializing StudyMate..."):
            st.session_state.studymate = load_studymate()
        st.session_state.messages = []
        st.session_state.pdf_processed = False
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None and not st.session_state.pdf_processed:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            with st.spinner("Processing PDF..."):
                try:
                    chunks = st.session_state.studymate.upload_and_process_pdf(tmp_path)
                    st.success(f"✅ Processed {chunks} text chunks")
                    st.session_state.pdf_processed = True
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        
        if st.session_state.pdf_processed:
            st.info("📚 PDF is loaded and ready for questions!")
        
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        if not st.session_state.pdf_processed:
            st.warning("⚠️ Please upload a PDF document first!")
            return
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.studymate.ask_question(prompt)
                
                st.markdown(response['answer'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"📊 Confidence: {response['confidence']:.2f}")
                with col2:
                    st.caption(f"⏱️ Time: {response['query_time']:.2f}s")
                with col3:
                    st.caption(f"📚 Sources: {len(response['sources'])}")
        
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})

if __name__ == "__main__":
    main()
