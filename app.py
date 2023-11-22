import os
from typing import List
import streamlit as st 
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings #, HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.srt import SRTLoader

from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader

from prompt_template_utils import get_prompt_template

DOCS_PATH="documents/context"
USER_IMAGES="documents/images/observer.png"
BRAVENE_IMAGES="documents/images/bravane.png"

loaders = {
    '.pdf': PyMuPDFLoader,
    '.csv': CSVLoader,
    '.doc*': UnstructuredWordDocumentLoader,
    '.txt': TextLoader,
    '.srt': SRTLoader,
}

def get_documents():
    documents: List[Document] = []
    # Create DirectoryLoader instances for each file type
    if(os.path.isdir(DOCS_PATH)):
        pdf_loader = create_directory_loader('.pdf', DOCS_PATH)
        csv_loader = create_directory_loader('.csv', DOCS_PATH)
        txt_loader = create_directory_loader('.txt', DOCS_PATH)
        doc_loader = create_directory_loader('.doc*', DOCS_PATH)
        srt_loader = create_directory_loader('.srt', DOCS_PATH)


        pdf_documents = pdf_loader.load()
        csv_documents = csv_loader.load()
        txt_documents = txt_loader.load()
        doc_documents = doc_loader.load()
        srt_documents = srt_loader.load()

        documents = csv_documents + txt_documents + doc_documents + pdf_documents + srt_documents
    else:
        os.mkdir("documents/context") 
    return documents


# Define a function to create a DirectoryLoader for a specific file type
def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders[file_type],
    )

def get_text_chunks(documents: List[Document]):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks: List[Document] = text_splitter.split_documents(documents)
    return chunks


def get_local_vectorstore(embeddings):
    if os.path.isdir("faiss_index"):
        vectorstore =  FAISS.load_local("faiss_index", embeddings)
        return vectorstore

def get_vectorstore(documents_chunks: List[Document], embeddings):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_documents(documents=documents_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore


def get_conversation_chain(vectorstore: FAISS):

    llm = ChatOpenAI( 
        temperature=0.7,
        model_name="gpt-3.5-turbo-0301",
    )

    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    prompt, memory = get_prompt_template( history=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs=dict(prompt=prompt)
    )

    return conversation_chain

def handle_userinput(user_question):

    # React to user input
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display user message in chat message container
    for i, history in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message(name="User", avatar=USER_IMAGES):
                content: str = history.content
                st.markdown(content)
        else :
            with st.chat_message("Bravene", avatar=BRAVENE_IMAGES):
                content = content.replace('[INST]','').replace('[SYS]','').replace('<<INST>>','').replace('<<SYS>>','').replace('Assistant:','')
                st.markdown(history.content)
    if not response:
        with st.chat_message("User"):
            st.markdown(user_question)

def main():
    load_dotenv()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "conversation" not in st.session_state:
            st.session_state.conversation = None
            with st.spinner("Processing"):
                embeddings = OpenAIEmbeddings()
            # create vector store
                vectorstore = get_local_vectorstore(embeddings)
                if(vectorstore):
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                else: 
                    st.session_state.conversation = None
                            
    st.header("braveneWWWorld Chat")

    with st.sidebar:
        if st.button("PROCESS DATA"):
            with st.spinner("Processing"):
                embeddings = OpenAIEmbeddings()
                # get pdf text
                documents: List[Document] = get_documents()
                
                if(documents.__len__() > 0):
                    # get the text chunks
                    documents_chunks: List[Document] = get_text_chunks(documents)

                    # create vector store
                    if(documents_chunks.__len__() > 0):
                        vectorstore = get_vectorstore(documents_chunks, embeddings)
                        if(vectorstore):
                            st.write("The data was processed successfully")
                            st.header("Processed Documents: ")
                            for doc in documents:
                                st.write(str(doc.metadata.get('source')).split('/')[2])
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    else: st.write("No document whit content.")
                else: st.write("No document to be processed.")
                
    if st.session_state.conversation:
        user_question = st.chat_input("Talk dirt to me!")
        with st.spinner("Processing"):
            if user_question:
                handle_userinput(user_question)

if __name__ == '__main__':
    main()
