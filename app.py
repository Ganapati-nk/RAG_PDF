# RAG QNA WITH CHAT HISTORY

import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()


os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')

os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



st.title("Conversational RAG with PDF and MSG History")
st.write("Upload Pdf and Chat with Content")

llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
session_id=st.text_input('Session_Id',value="default_session")

if 'store' not in st.session_state:
    st.session_state.store={}
uploaded_Files=st.file_uploader("Choose a PDF File",type='pdf',accept_multiple_files=True)
if uploaded_Files:
    documents=[]
    for uploader_file in uploaded_Files:
        temppdf=f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploader_file.getvalue())
            file_name=uploader_file.name
        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
    if os.path.exists("./temp.pdf"):
        os.remove("./temp.pdf")
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
    splits=text_splitter.split_documents(docs)
    vector_store=FAISS.from_documents(documents=splits,embedding=embeddings)
    retriever=vector_store.as_retriever()


    context_system_prompt=(
        "Given the chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a stand alone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )


    context_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    history_aware_retriever1=create_history_aware_retriever(llm,retriever,context_q_prompt)



    system_prompt=(
        "you are an assistant for question answering task. "
        "Use the following retrived Context to answer "
        "the question. if you dont know the answer say that "
        "you dont know. Use three sentences maximum and"
        "keep answer concise. "
        "\n\n"
        "{context}"

    )

    qa_prompt=ChatPromptTemplate.from_messages(
        [
            ('system',system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human',"{input}"),
        ]

    )

    qa_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever1,qa_chain)


    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    


    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'

    )


    user_input=st.text_input("Your Question:")
    if user_input:
        session_history=get_session_history(session_id)
        response=conversational_rag_chain.invoke(
            {'input':user_input},
            config={"configurable":{"session_id":session_id},
                    }
        )

        # st.write(st.session_state.store)
        st.write("Assistant:",response['answer'])
        # st.write("chat_history:",session_history.messages)