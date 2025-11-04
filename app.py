import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
#from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import time
import shutil
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
import faiss

import nest_asyncio
nest_asyncio.apply()

load_dotenv()

INDEX_DIRECTORY = "faiss_index"
PDF_DIRECTORY = "pdfs"


# def create_embeddings_from_pdfs():
#     if not os.path.exists(PDF_DIRECTORY) or not os.listdir(PDF_DIRECTORY):
#         st.warning(f"'{PDF_DIRECTORY}' folder not found or there are no PDF files inside. Please add your PDFs to this folder.")
#         return False
#     with st.spinner(f"Creating embeddings from PDFs in the '{PDF_DIRECTORY}' folder... This may take some time."):
#         try:
#             loader = DirectoryLoader(PDF_DIRECTORY, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=False)
#             documents = loader.load()
#             if not documents:
#                 st.error("Failed to load PDF files or their contents are empty.")
#                 return False
#         except Exception as e:
#             st.error(f"An error occurred while loading PDF files: {e}")
#             return False
#
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=200,
#             length_function=len,
#             separators=["\n\n", "\n", " ", ""]
#         )
#         texts = text_splitter.split_documents(documents)
#         if not texts:
#             st.error("Could not extract text from PDFs.")
#             return False
#
#         try:
#             embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#             db = FAISS.from_documents(texts, embeddings)
#             db.save_local(INDEX_DIRECTORY)
#             st.success("Embeddings were successfully created and saved!")
#         except Exception as e:
#             st.error(f"An error occurred while creating or saving embeddings: {e}")
#             return False
#     return True

class DummyEmbeddings(Embeddings):
    """API çağrısı yapmayan dummy embeddings sınıfı - sadece mevcut index'i yüklemek için"""
    def embed_documents(self, texts):
        # Mevcut index'i yüklemek için dummy embedding döndür
        # Gerçek embedding'ler zaten index'te mevcut
        return [[0.0] * 768 for _ in texts]  # Google embedding boyutu genellikle 768
    
    def embed_query(self, text):
        # Mevcut index'i yüklemek için dummy embedding döndür
        return [0.0] * 768

def get_rag_chain():
    index_file_path = os.path.join(INDEX_DIRECTORY, "index.faiss")
    if not os.path.exists(index_file_path):
        st.error(f"FAISS index not found at {index_file_path}.")
        return None

    try:
        # API çağrısı yapmayan dummy embeddings kullan
        # Mevcut index'teki embedding'ler zaten mevcut
        embeddings = DummyEmbeddings()
        db = FAISS.load_local(INDEX_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}.")
        return None

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "lambda_mult": 0.7,
            "fetch_k": 20
        }
    )

    system_message_content = """Sen Dokuz Eylül Üniversitesi'nin dijital asistanısın. Görerin sadece verilen "Context" bölümündeki bilgileri kullanarak kullanıcının sorularını yanıtlamak.

ÖNEMLİ KURALLAR:
1. SADECE verilen "Context" bölümündeki bilgileri kullan. Kendi bilgilerini ASLA ekleme.
2. Context'te yeterli bilgi yoksa, kibarca bilginin mevcut olmadığını belirt. TAHMİN YAPMA.
3. Kullanıcının sorusunu dikkatlice analiz et ve context'teki ilgili bilgileri kapsamlı şekilde yanıtla.
4. Yanıtını kullanıcının sorusunun dilinde ver:
    - Soru Türkçe ise, Türkçe yanıtla
    - Soru İngilizce ise, İngilizce yanıtla
    - Dil tespit edemezsen, Türkçe kullan
5. "Sen kimsin?" tarzı sorularda şu yanıtı ver:
    * Türkçe: "Ben Dokuz Eylül Üniversitesi'nin dijital asistanıyım. Üniversitemizle ilgili sorularınıza yardımcı olmak için buradayım. Size nasıl yardımcı olabilirim?"
    * İngilizce: "I am the digital assistant of Dokuz Eylül University. I'm here to help with your questions about our university. How can I assist you?"
6. Yanıtlarını Markdown formatında ve açık paragraflar halinde sun.
7. Context'teki bilgileri doğrudan kullan, kendi yorumunu katma."""

    human_message_template_str = """Context:
{context}

Soru: {input}

Yanıt:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message_content),
        ("human", human_message_template_str)
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        convert_system_message_to_human=True
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def main():
    st.set_page_config(page_title="DEUbot | DEU Assistant", page_icon="🎓", layout="wide")

    if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
        st.error("GOOGLE_API_KEY not found or empty in the .env file.")
        st.markdown("You can get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
        st.stop()

    st.markdown("""
    <style>
        .stChatInputContainer > div > div > textarea { background-color: #ffffff; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # Session state init
    if "conversations" not in st.session_state or not isinstance(st.session_state.conversations, dict):
        st.session_state.conversations = {}
    if "active_conversation_id" not in st.session_state:
        st.session_state.active_conversation_id = None
    if "next_conversation_id" not in st.session_state:
        valid_ids = [int(k) for k in st.session_state.conversations.keys() if str(k).isdigit()] if st.session_state.conversations else []
        st.session_state.next_conversation_id = max(valid_ids) + 1 if valid_ids else 1
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    with st.sidebar:
        st.image("Deu_logo.png", width=100)
        st.title("DEUbot")
        st.caption("Dokuz Eylül University Assistant")
        st.markdown("---")

        if st.button("💬 Start New Chat", use_container_width=True, type="primary"):
            conv_id = st.session_state.next_conversation_id
            st.session_state.conversations[conv_id] = {"title": f"Chat {conv_id}", "messages": []}
            st.session_state.active_conversation_id = conv_id
            st.session_state.next_conversation_id += 1
            st.rerun()

        st.markdown("#### Chat History")
        if not st.session_state.conversations:
            st.caption("No chats started yet.")
        else:
            sorted_conv_ids = sorted([int(k) for k in st.session_state.conversations.keys() if str(k).isdigit()], reverse=True)
            for conv_id in sorted_conv_ids:
                conv = st.session_state.conversations.get(conv_id)
                if conv:
                    button_label = conv.get('title', f"Chat {conv_id}")
                    if st.button(f"{button_label}", key=f"conv_{conv_id}", use_container_width=True,
                                 type="secondary" if st.session_state.active_conversation_id == conv_id else "tertiary"):
                        st.session_state.active_conversation_id = conv_id
                        st.rerun()

        st.markdown("---")
        index_file_path = os.path.join(INDEX_DIRECTORY, "index.faiss")
        if not os.path.exists(index_file_path):
            st.warning("Database (FAISS index) not found. Please ensure index.faiss exists in the faiss_index folder.")
        else:
            st.success("Database found. Using existing FAISS index.")

    if st.session_state.active_conversation_id is None and st.session_state.conversations:
        valid_ids = [int(k) for k in st.session_state.conversations.keys() if str(k).isdigit()]
        if valid_ids:
            st.session_state.active_conversation_id = max(valid_ids)

    active_conv_id = st.session_state.active_conversation_id
    current_conv_data = st.session_state.conversations.get(active_conv_id) if active_conv_id else None

    if current_conv_data:
        st.header(f"💬 {current_conv_data.get('title', f'Chat {active_conv_id}')}")
        for message in current_conv_data.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type your message..."):
            current_conv_data.setdefault("messages", []).append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.rag_chain is None:
                rag_chain = get_rag_chain()
                if rag_chain is None:
                    st.error("RAG chain could not be created. Please check configuration and API key.")
                    st.stop()
                st.session_state.rag_chain = rag_chain
            else:
                rag_chain = st.session_state.rag_chain

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("DEUbot is thinking..."):
                    try:
                        response = rag_chain.invoke({"input": prompt})
                        answer = response.get("answer", "No answer received.")
                    except Exception as e:
                        st.error(f"An error occurred while receiving the answer: {e}")
                        answer = "Üzgünüm, bir hata oluştu ve şu anda yanıt veremiyorum."

                full_response = ""
                if isinstance(answer, str):
                    for chunk in answer.split():
                        full_response += chunk + " "
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.03)
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.markdown("Beklenmeyen formatta yanıt alındı.")       
            # Add assistant response to history
            current_conv_data.setdefault("messages", []).append({"role": "assistant", "content": answer})
            st.session_state.conversations[active_conv_id] = current_conv_data
    else:
        st.info("👋 Merhaba! Ben DEUbot. Dokuz Eylül Üniversitesi hakkında bilgi almak için kenar çubuğundan 'Yeni Sohbet Başlat' butonuna tıklayın veya mevcut bir sohbet seçin.")
        if not os.path.exists(PDF_DIRECTORY) or not os.listdir(PDF_DIRECTORY):
            st.warning(f"'{PDF_DIRECTORY}' klasöründe PDF dosyası bulunmuyor. Lütfen PDF'lerinizi ekleyin.")
        if not os.path.exists(INDEX_DIRECTORY):
            st.warning("Veritabanı (FAISS index) henüz oluşturulmamış. PDF'lerinizden veritabanı oluşturmak için kenar çubuktaki 'Veritabanı Oluştur' butonunu kullanın.")

if __name__ == "__main__":
    main()
