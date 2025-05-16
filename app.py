import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import time
import shutil

# New imports
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()

INDEX_DIRECTORY = "faiss_index"
PDF_DIRECTORY = "pdfs"

# Create FAISS index from PDF files
def create_embeddings_from_pdfs():
    if not os.path.exists(PDF_DIRECTORY) or not os.listdir(PDF_DIRECTORY):
        st.warning(f"'{PDF_DIRECTORY}' folder not found or there are no PDF files inside. Please add your PDFs to this folder.")
        return False
    with st.spinner(f"Creating embeddings from PDFs in the '{PDF_DIRECTORY}' folder... This may take some time."):
        try:
            # use_multithreading=True can sometimes cause issues on Windows, set to False or remove if needed.
            loader = DirectoryLoader(PDF_DIRECTORY, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=False)
            documents = loader.load()
            if not documents:
                st.error("Failed to load PDF files or their contents are empty.")
                return False
        except Exception as e:
            st.error(f"An error occurred while loading PDF files: {e}")
            return False

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        if not texts:
            st.error("Could not extract text from PDFs.")
            return False

        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(INDEX_DIRECTORY)
            st.success("Embeddings were successfully created and saved!")
        except Exception as e:
            st.error(f"An error occurred while creating or saving embeddings: {e}")
            return False
    return True

# Create conversation chain with retriever
def get_chain(chat_memory=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists(INDEX_DIRECTORY):
        st.error("FAISS index file not found. Please create embeddings first.")
        return None, None
    try:
        db = FAISS.load_local(INDEX_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}. The index file might be corrupted. Try recreating it from the sidebar.")
        return None, None

    retriever = db.as_retriever(search_kwargs={"k": 5}) # Increase k to get more relevant documents

    if chat_memory is None:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    else:
        memory = chat_memory

    system_message_content = """You are an assistant bot providing information about Dokuz Eylül University. Your task is to answer the user's questions ONLY using the provided text excerpts.

RULES:
1. ALWAYS base your answers on the information in the provided "Context" section. DO NOT USE information outside the context.
2. If the context does not contain enough information to answer the question, politely state that the information is not available or that you cannot assist. DO NOT GUESS.
3. Carefully analyze the user's question and the relevant text. Provide a detailed and comprehensive answer.
4. Generate your answers in the language the user asked the question in (Turkish, English, etc.).
For example:
   - If the question is in Turkish, answer in Turkish.
   - If the question is in English, answer in English.
   - If the question is in any other language, answer in that language.
   - If you cannot detect the language clearly, default to Turkish.
5. If the user asks "Who are you?", "What are you?" or similar questions, respond as follows:
    * In Turkish: "Ben Dokuz Eylül Üniversitesi'nin dijital asistanıyım. Üniversitemizle ilgili sorularınıza yardımcı olmak için buradayım. Size nasıl yardımcı olabilirim?"
    * In English: "I am the digital assistant of Dokuz Eylül University. I'm here to help with your questions about our university. How can I assist you?"
6. Remember previous conversations and shape your answers accordingly.
7. Present your answers in Markdown format and in clear paragraphs."""

    human_message_template_str = """
Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:"""

    custom_chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message_content),
        HumanMessagePromptTemplate.from_template(human_message_template_str)
    ])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, convert_system_message_to_human=True)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_chat_prompt},
        return_source_documents=True, # Set to True if you want to see source documents
        verbose=False
    )
    
    return chain, memory

def main():
    st.set_page_config(page_title="DEUbot | DEU Assistant", page_icon="🎓", layout="wide")

    if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
        st.error("GOOGLE_API_KEY not found or empty in the .env file. Please add your API key and make sure it is valid.")
        st.markdown("You can get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
        st.stop()

    st.markdown("""
    <style>
        .stApp {
            /* background-color: #f0f2f6; */
        }
        .stChatInputContainer > div > div > textarea {
            background-color: #ffffff;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    if "conversations" not in st.session_state or not isinstance(st.session_state.conversations, dict):
        st.session_state.conversations = {}

    if "active_conversation_id" not in st.session_state:
        st.session_state.active_conversation_id = None

    if "next_conversation_id" not in st.session_state:
        if st.session_state.conversations:
            valid_ids = [int(k) for k in st.session_state.conversations.keys() if str(k).isdigit()]
            st.session_state.next_conversation_id = max(valid_ids) + 1 if valid_ids else 1
        else:
            st.session_state.next_conversation_id = 1

    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "memory" not in st.session_state:
        st.session_state.memory = None

    with st.sidebar:
        st.image("Deu_logo.png", width=100)
        st.title("DEUbot")
        st.caption("Dokuz Eylül University Assistant")
        st.markdown("---")

        if st.button("💬 Start New Chat", use_container_width=True, type="primary"):
            conv_id = st.session_state.next_conversation_id
            st.session_state.conversations[conv_id] = {
                "title": f"Chat {conv_id}",
                "messages": [],
                "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
            }
            st.session_state.active_conversation_id = conv_id
            st.session_state.next_conversation_id += 1
            st.session_state.chain = None
            st.session_state.memory = st.session_state.conversations[conv_id]["memory"]
            st.rerun()

        st.markdown("#### Chat History")
        if not st.session_state.conversations:
            st.caption("No chats started yet.")
        else:
            try:
                sorted_conv_ids = sorted([int(k) for k in st.session_state.conversations.keys() if str(k).isdigit()], reverse=True)
            except ValueError:
                st.warning("There is an inconsistency in chat IDs.")
                sorted_conv_ids = []

            for conv_id in sorted_conv_ids:
                conv = st.session_state.conversations.get(conv_id)
                if conv:
                    button_label = conv.get('title', f"Chat {conv_id}")
                    if st.button(f"{button_label}", key=f"conv_{conv_id}", use_container_width=True,
                                 type="secondary" if st.session_state.active_conversation_id == conv_id else "tertiary"):
                        st.session_state.active_conversation_id = conv_id
                        st.session_state.memory = conv["memory"]
                        st.session_state.chain = None
                        st.rerun()
        st.markdown("---")
        if not os.path.exists(INDEX_DIRECTORY):
            st.warning("Database (index) not found.")
            if st.button("📚 Create Database", use_container_width=True):
                if create_embeddings_from_pdfs():
                    st.success("Database created successfully!")
                    st.rerun()
                else:
                    st.error("Database creation failed. Please check the PDF folder and API key.")
        else:
            if st.button("🔄 Update Database", use_container_width=True):
                if create_embeddings_from_pdfs():
                    st.success("Database updated successfully!")
                    st.rerun()
                else:
                    st.error("Database update failed.")

    if st.session_state.active_conversation_id is None and st.session_state.conversations:
        try:
            valid_ids = [int(k) for k in st.session_state.conversations.keys() if str(k).isdigit()]
            if valid_ids:
                st.session_state.active_conversation_id = max(valid_ids)
        except ValueError:
            pass

    active_conv_id = st.session_state.active_conversation_id
    current_conv_data = None

    if active_conv_id is not None and active_conv_id in st.session_state.conversations:
        current_conv_data = st.session_state.conversations[active_conv_id]

    if current_conv_data:
        conv_title = current_conv_data.get("title", f"Chat {active_conv_id}")
        st.header(f"💬 {conv_title}")

        for message in current_conv_data.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(f"Write your message in {conv_title}..."):
            if not os.path.exists(INDEX_DIRECTORY):
                st.error("Please create the database from the sidebar first.")
                st.stop()

            current_conv_data.setdefault("messages", []).append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.chain is None or st.session_state.memory != current_conv_data["memory"]:
                chain, memory = get_chain(current_conv_data["memory"])
                if chain is None:
                    st.error("Chat chain could not be created. Please check configuration and API key.")
                    st.stop()
                st.session_state.chain = chain
                st.session_state.memory = memory
            else:
                chain = st.session_state.chain

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("DEUbot is thinking..."):
                    try:
                        response = chain.invoke({"question": prompt})
                        answer = response.get("answer", "No answer received.")
                        # source_documents = response.get("source_documents") 
                        # if source_documents:
                        # with st.expander("Sources"):
                        # for doc in source_documents:
                        # st.caption(f"Page: {doc.metadata.get('page', 'Unknown')} - Content: {doc.page_content[:100]}...")
                    except Exception as e:
                        st.error(f"An error occurred while receiving the answer: {e}")
                        answer = "Sorry, I encountered an issue and cannot respond right now."

                full_response = ""
                if isinstance(answer, str):
                    for chunk in answer.split():
                        full_response += chunk + " "
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.05)
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.markdown("Received response in an unexpected format.")

            current_conv_data.setdefault("messages", []).append({"role": "assistant", "content": answer})
            st.session_state.conversations[active_conv_id] = current_conv_data
    else:
        st.info("👋 Hello! I'm DEUbot. To get information about Dokuz Eylül University, please click 'Start New Chat' in the sidebar or select an existing chat.")
        if not os.path.exists(PDF_DIRECTORY) or not os.listdir(PDF_DIRECTORY):
            st.warning(f"There are no PDF files in the '{PDF_DIRECTORY}' folder. Please add your PDFs.")
        if not os.path.exists(INDEX_DIRECTORY):
            st.warning("Database (FAISS index) has not been created yet. Please use the 'Create Database' button in the sidebar to generate a database from your PDFs.")

if __name__ == "__main__":
    main()