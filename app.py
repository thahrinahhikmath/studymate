import streamlit as st
import os
import time
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import tempfile

# StudyMate Classes
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
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
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
    def __init__(self):
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            try:
                self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            except Exception:
                st.error("Could not load embedding model. Please check your internet connection.")
                return
        
        self.index = None
        self.texts = []
        
    def add_texts(self, texts: List[str]):
        """Add texts to vector store"""
        if not texts:
            return
            
        try:
            self.texts.extend(texts)
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
        except Exception as e:
            st.error(f"Error processing text chunks: {e}")
        
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar texts"""
        if self.index is None or len(self.texts) == 0:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
            faiss.normalize_L2(query_embedding)
            
            k = min(k, len(self.texts))
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.texts):
                    results.append({
                        'text': self.texts[idx],
                        'score': float(score),
                        'index': int(idx)
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Error searching documents: {e}")
            return []

class StudyMate:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        
        # Initialize QA pipeline
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                return_tensors="pt"
            )
        except Exception as e:
            st.error(f"Could not load QA model: {e}")
            self.qa_pipeline = None
        
    def upload_and_process_pdf(self, pdf_path: str):
        """Process uploaded PDF"""
        try:
            # Extract text
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            if not text.strip():
                raise ValueError("No readable text found in PDF")
            
            # Chunk text
            chunks = self.pdf_processor.chunk_text(text)
            if not chunks:
                raise ValueError("Could not process PDF text")
            
            # Add to vector store
            self.vector_store.add_texts(chunks)
            
            return len(chunks)
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return 0
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Answer question based on uploaded documents"""
        start_time = time.time()
        
        if not self.qa_pipeline:
            return {
                'answer': "QA model not available. Please try refreshing the app.",
                'confidence': 0.0,
                'sources': [],
                'query_time': time.time() - start_time
            }
        
        try:
            # Search for relevant context
            search_results = self.vector_store.search(question, k=3)
            
            if not search_results:
                return {
                    'answer': "No relevant information found in the uploaded document. Try asking about topics covered in your PDF.",
                    'confidence': 0.0,
                    'sources': [],
                    'query_time': time.time() - start_time
                }
            
            # Combine context from search results
            context = " ".join([result['text'] for result in search_results[:2]])
            
            # Truncate context to avoid token limits
            if len(context.split()) > 300:
                context = " ".join(context.split()[:300])
            
            # Get answer from QA model
            result = self.qa_pipeline(question=question, context=context)
            
            return {
                'answer': result['answer'],
                'confidence': result['score'],
                'sources': search_results[:2],
                'query_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'answer': f"Sorry, I encountered an error while processing your question. Please try rephrasing it.",
                'confidence': 0.0,
                'sources': [],
                'query_time': time.time() - start_time
            }

# Streamlit App Configuration
st.set_page_config(
    page_title="StudyMate", 
    page_icon="🎓", 
    layout="wide"
)

# Cache the StudyMate instance
@st.cache_resource
def get_studymate():
    return StudyMate()

def reset_session():
    """Reset the session state"""
    keys_to_keep = ['studymate_instance']  # Keep the cached model
    keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
    for key in keys_to_delete:
        del st.session_state[key]

def main():
    # Title
    st.title("🎓 StudyMate - AI-Powered PDF Q&A")
    st.markdown("Upload your PDF documents and ask questions!")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'studymate_instance' not in st.session_state:
        with st.spinner("🚀 Initializing StudyMate..."):
            st.session_state.studymate_instance = get_studymate()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Documents")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            if not st.session_state.pdf_processed:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                # Process the PDF
                with st.spinner("📚 Processing PDF... This may take a minute."):
                    try:
                        chunks = st.session_state.studymate_instance.upload_and_process_pdf(tmp_path)
                        
                        if chunks > 0:
                            st.success(f"✅ Successfully processed {chunks} text chunks!")
                            st.session_state.pdf_processed = True
                        else:
                            st.error("❌ Could not process the PDF. Please try a different file.")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                    
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
            else:
                st.success("📄 PDF loaded successfully!")
        
        # Show status
        if st.session_state.pdf_processed:
            st.info("🤖 Ready to answer questions!")
        else:
            st.warning("⚠️ Please upload a PDF to get started")
        
        # Reset button
        if st.button("🔄 Reset Session"):
            reset_session()
            # Force refresh without using experimental functions
            st.session_state.pdf_processed = False
            st.session_state.messages = []
    
    # Main content area
    if not st.session_state.pdf_processed:
        st.info("👆 Please upload a PDF document using the sidebar to begin!")
        return
    
    # Chat interface
    st.subheader("💬 Ask Questions About Your Document")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                st.write(message["content"])
                if "metadata" in message:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"📊 Confidence: {message['metadata'].get('confidence', 0):.2f}")
                    with col2:
                        st.caption(f"⏱️ Response time: {message['metadata'].get('query_time', 0):.1f}s")
                    with col3:
                        st.caption(f"📄 Sources used: {message['metadata'].get('sources', 0)}")
    
    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.studymate_instance.ask_question(prompt)
                
                # Display answer
                st.write(response['answer'])
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"📊 Confidence: {response['confidence']:.2f}")
                with col2:
                    st.caption(f"⏱️ Response time: {response['query_time']:.1f}s")
                with col3:
                    st.caption(f"📄 Sources used: {len(response['sources'])}")
        
        # Add assistant message to session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response['answer'],
            "metadata": {
                "confidence": response['confidence'],
                "query_time": response['query_time'],
                "sources": len(response['sources'])
            }
        })

if __name__ == "__main__":
    main()
