import streamlit as st
import os
import tempfile
from studymate import StudyMate

st.set_page_config(
    page_title="StudyMate - AI PDF Q&A", 
    page_icon="🎓", 
    layout="wide"
)

def main():
    st.title("🎓 StudyMate - AI-Powered PDF Q&A")
    st.markdown("Upload your PDF documents and ask questions!")
    
    # Initialize StudyMate
    if 'studymate' not in st.session_state:
        with st.spinner("Initializing StudyMate..."):
            st.session_state.studymate = StudyMate()
        st.session_state.messages = []
        st.session_state.pdf_processed = False
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None and not st.session_state.pdf_processed:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            # Process PDF
            with st.spinner("Processing PDF..."):
                try:
                    chunks = st.session_state.studymate.upload_and_process_pdf(tmp_path)
                    st.success(f"✅ Processed {chunks} text chunks")
                    st.session_state.pdf_processed = True
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                finally:
                    # Clean up
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        
        if st.session_state.pdf_processed:
            st.info("📚 PDF is loaded and ready for questions!")
        
        # Reset button
        if st.button("🔄 Reset Session"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "metadata" in message:
                st.markdown(message["content"])
                # Show metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"📊 Confidence: {message['metadata']['confidence']:.2f}")
                with col2:
                    st.caption(f"⏱️ Time: {message['metadata']['query_time']:.2f}s")
                with col3:
                    st.caption(f"📚 Sources: {message['metadata']['sources']}")
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        if not st.session_state.pdf_processed:
            st.warning("⚠️ Please upload a PDF document first!")
            return
            
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.studymate.ask_question(prompt)
                
                st.markdown(response['answer'])
                
                # Show additional info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"📊 Confidence: {response['confidence']:.2f}")
                with col2:
                    st.caption(f"⏱️ Time: {response['query_time']:.2f}s")
                with col3:
                    st.caption(f"📚 Sources: {len(response['sources'])}")
        
        # Add assistant response
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
