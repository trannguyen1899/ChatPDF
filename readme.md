## Introduction
------------
 ChatPDF is a Python application that allows you to chat with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.



## How It Works
------------
![ChatPDF App Diagram](diagram_flow.PNG)

The application follows these steps to provide responses to your questions:

1. PDF Loading: The app reads multiple PDF documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.



## Dependencies and Installation
----------------------------
To install the ChatPDF App, please follow these steps:

1. Clone the repository to your local machine.

2. Create a virtual environment with **python version 3.10.0**:
   ```
   conda create -n my_env python==3.10.0
   conda activate my_env
   ```
3. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

4. If you intend to use Llama, download Llama LLM (model_basename: "llama-2-7b-chat.ggmlv3.q8_0.bin") 
by running llama.py. Llama model will be downloaded into the folder *model*. 
Please note that the size of the model is up to 7GB.
   ```
   python3 download_llama.py
   ```

*Note*: If you have **problems** with building wheel when **installing llama-cpp library**, you can try installing it by compiling from the source (instruction in this link https://python.langchain.com/docs/integrations/llms/llamacpp, section "Installation with Windows")



## Usage
-----
To use the ChatPDF App, follow these steps:

1. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run main.py
   ```

2. The application will launch in your default web browser, displaying the user interface.

3. Load multiple PDF documents into the app by following the provided instructions.

4. Ask questions in natural language about the loaded PDFs using the chat interface.



## Contributing
------------
This repository is intended for educational purposes.
