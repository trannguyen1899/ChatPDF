import streamlit as st

def sidebar():
    with st.sidebar:
        
        st.markdown(
            "## How to use\n"
            "1. Choose a LLM model"
            "2. Upload a pdf, docx, or txt file📄\n"
            "3. Ask a question about the document💬\n"
        )

        st.markdown(        
            "***Guideness for choosing an approriate LLM model***:\n"
            "- Choose **Llama** if you prioritize security and confidentiality of your documents, but it would take a while to respond.\n"
            "- Choose **GPT 3.5 turbo** if you would like an instant response, but it requires an OpenAI API key"
        )
        
        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "📖chatPDF allows you to ask questions about your "
            "documents and get accurate answers."
        )
        st.markdown(
            "This tool is a work in progress. "
            "with your feedback and suggestions💡"
        )
        st.markdown("---")
