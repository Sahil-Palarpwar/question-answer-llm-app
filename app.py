from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import HuggingFaceHub
import gradio as gr
from langchain.chains import RetrievalQAWithSourcesChain
import os

def answer_question(api_key, pdf_path, question):
    # Set the API key as an environment variable
    os.environ["OPENAI_API_KEY"] = api_key

    # Open the PDF file
    # with open(pdf_path, 'r', encoding='utf-8') as file:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(pages, embeddings)

    model = OpenAI()
    qa_chain = load_qa_with_sources_chain(model, chain_type="stuff")
    qa = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=db.as_retriever())

    ans = qa(question)

    return ans['answer']

# Create the interface
api_key_input = gr.inputs.Textbox(label="API Key", type='password')
pdf_path_input = gr.inputs.Textbox(label="PDF File Path")
question_input = gr.inputs.Textbox(label="Question")
output = gr.outputs.Textbox(label="Answer")

iface = gr.Interface(fn=answer_question, inputs=[api_key_input, pdf_path_input, question_input], outputs=output,
                     title="PDF Question-Answering")

# Run the interface
iface.launch(debug=True)
