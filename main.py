import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore.document import Document

from htmlTemplates import css, bot_template, user_template
from sidebar import sidebar

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGML"
#model_basename = "llama-2-7b-chat.ggmlv3.q8_0.bin"
#model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

def read_pdf(pdf_docs) -> list[Document]: 
    """ READ THE CONTENTS OF UPLOADED DOCUMENTS """
    docs = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        doc = Document(
            page_content = text,
            metadata={
                "source": f"{pdf.name}"
            }
        )   
        docs.append(doc)     
    return docs


def get_chunks(docs) -> list[Document]:
    """ CHUNKING """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def get_vectorstore(document_chunks, EMBEDDING):
    """ CREATE VECTOR STORE """
    vectorstore = FAISS.from_documents(documents=document_chunks, embedding=EMBEDDING)
    return vectorstore


def get_conversation_chain(LLM, vectorstore):
    """ BUILD CONVERSATION CHAIN """
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      input_key='question', 
                                      output_key='answer', 
                                      return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain


def handle_query_input(user_question):
    """
    GENERATE RESPONSE
    Return source documents and chat history
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    if response['source_documents']:
        st.session_state.source = response['source_documents']        




def main():
    
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)


    # INITIALIZE st.session_state:
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'LLM_type' not in st.session_state:
        st.session_state.LLM_type = None
    if 'source' not in st.session_state:
        st.session_state.source = None
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.0
    
    # ============================================================================================
    # MAIN LAYOUT
    # This area display the question input widget + the conversation
    # After inputing the question, user MUST click button "Go" to start generating the response
    # ============================================================================================
    st.header("Chat with multiple PDFs :books:")

    # QUERY INPUT   
    container = st.container()
    col1, col2 = st.columns([0.9, 0.1]) # A text input widget and a button Go to enter the question
    with col1:
        user_question = st.text_input("Ask a question about your documents:")
    with col2:
        st.markdown('<p style="font-family:Courier; color:White; font-size: 8px;">H</p>', 
                    unsafe_allow_html=True)
        if st.button('Go'):
            handle_query_input(user_question) # generate the response

    # DISPLAY CONVERSATION
    if st.session_state.chat_history != None:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                container.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                container.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)



    #========================================
    #     CUSTOM SIDEBAR
    #========================================
    with st.sidebar:
        
        # CHOOSE A LLM
        st.subheader("1. Choose your LLM")

        def set_state(i):
            st.session_state.stage = i

        st.button('Llama 2', on_click=set_state, args=[2])
        st.button('GPT 3.5 turbo', on_click=set_state, args=[1])
    
        if st.session_state.stage == 1: # if "GPT 3.5 turbo"
            key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="Paste your OpenAI API key here (sk-...)",
                help="You can get your API key from https://platform.openai.com/account/api-keys.",
            )
            st.session_state.api_key = key
            st.session_state.LLM_type = 'openai'
            with st.sidebar:
                st.markdown('OpenAI is chosen')

        if st.session_state.stage == 2: # if "Llama"
            st.session_state.LLM_type = 'llama'
            with st.sidebar:
                st.markdown('Llama is chosen')

        
        # UPLOAD DOCUMENTS AND PROCESS
        # If the user click on the Process button, the documents will be read, chunked and embedded.
        st.subheader("2. Upload your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        openai_api_key = st.session_state.api_key        
        if st.button("Process"): 
            with st.spinner("Processing"):
                # check embedding and LLM options:
                print(st.session_state.LLM_type)
                if st.session_state.LLM_type == 'openai':
                    EMBEDDING = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    LLM = ChatOpenAI(model="gpt-3.5-turbo", 
                                     temperature = st.session_state.temperature, 
                                     openai_api_key=openai_api_key)
                elif st.session_state.LLM_type == 'llama':
                    EMBEDDING = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                    model_path = "./model/llama-2-7b-chat.ggmlv3.q8_0.bin"
                    LLM = LlamaCpp(model_path=model_path,
                                   max_tokens=256,
                                   temperature=st.session_state.temperature,
                                   callback_manager=callback_manager,
                                   n_ctx=2048, verbose=False,)

                # get pdf text
                raw_docs = read_pdf(pdf_docs)

                # get the document chunks
                doc_chunks = get_chunks(raw_docs)

                # create vector store
                vectorstore = get_vectorstore(doc_chunks, EMBEDDING)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(LLM, vectorstore)


        # CUSTOMIZE THE TEMPERATURE. The default temp = 0 
        # To start applying the chosen temperature, users MUST click the button "Apply",
        # otherwise, the temperature won't change
        st.subheader("Set up the temperature")
        st.write("The default temperature is 0")
        temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        if st.button("Apply"):
            st.session_state.temperature = temperature


        # HOW TO USE, note on HOW TO CHOOSE APPRORIATE LLM, ABOUT US
        sidebar()

    # =======================================================
    #   SHOW 5 DOCUMENT SOURCES. This is a toggle button.
    # =======================================================
    with st.expander("Show source documents"):
        if st.session_state.source:
            for i in range(len(st.session_state.source)):
                if i == 6:
                    break
                st.subheader(f"Source {i+1}:")
                st.markdown(st.session_state.source[i].page_content)
                print()
                st.markdown(f"**From**: {st.session_state.source[i].metadata['source']}")
        else:
            st.markdown('No source document found')


if __name__ == '__main__':
    main()
