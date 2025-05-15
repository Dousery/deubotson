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

load_dotenv()

INDEX_DIRECTORY = "faiss_index"
PDF_DIRECTORY = "pdfs"

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyasına API anahtarınızı ekleyin.")

# Create FAISS index from PDF files
def create_embeddings_from_pdfs():
    loader = DirectoryLoader(PDF_DIRECTORY, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(INDEX_DIRECTORY)

# Create conversational chain with retriever
def get_chain(chat_memory=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(INDEX_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    # If chat_memory is None, create a new memory
    if chat_memory is None:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    else:
        memory = chat_memory

    prompt_template = """
     1. Sana kullanıcının sorusunu ve ilgili metin alıntılarını sağlayacağım.
     2. Görevin, yalnızca sağlanan metin alıntılarını kullanarak Dokuz Eylül Üniversitesi adına detaylı bir şekilde cevap vermektir.
     3. Yanıtı oluştururken şu kurallara dikkat et:
       - Sağlanan metin alıntısında açıkça yer alan bilgileri kullan.
       - Metin alıntısında açıkça bulunmayan cevapları tahmin etmeye veya uydurmaya çalışma.
       - Soruyu ve ilgili metni dikkatlice analiz et ve her iki konuda da ayrıntılı, kapsamlı bir cevap ver.
     4. Yanıtı, kullanıcının soru yazarken kullandığı dille oluştur.
     5. Kullanıcıya her zaman yardımcı olmaya çalış, ancak mevcut bilgilere dayanmayan yanıtlardan kaçın.
     6. Eğer "Sen kimsin" diye bir soru gelirse, kullanıcı hangi dili kullandıysa o dilde şu şekilde yanıt ver:
       - Türkçe: "Ben Dokuz Eylül Üniversitesinin asistan botuyum, amacım sizlere üniversite hakkında bilgi sağlamak! Size nasıl yardımcı olabilirim?"
       - İngilizce: "I am the assistant bot of Dokuz Eylül University, my purpose is to provide you with information about the university! How can I assist you?"
     7. Eğer kullanıcı daha önce bir kişi veya konu hakkında soru sorduysa, bu sorunun bağlamını hatırla ve önceki cevaplar ve sorularla bağlantılı olarak yanıt ver.

     Context:\n{context}\n
     Soru:\n{question}\n
     Yanıt:
    """

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )
    return chain, memory

# Display text with typing effect
def display_text_with_effect(text):
    placeholder = st.empty()
    displayed_text = ""
    for token in text.split():
        displayed_text += token + " "
        placeholder.markdown(f"<div style='font-size:15px;'>{displayed_text}</div>", unsafe_allow_html=True)
        time.sleep(0.05)

def main():
    st.set_page_config(page_title="DEUbot", page_icon="🎓", layout="centered")

    st.header("Hi I'm DEUbot 🤖")
    st.subheader("What would you like to learn about our university?")

    # Start Streamlit session_state 
    if "conversations" not in st.session_state:
        st.session_state.conversations = []  # A list to hold conversations 
    if "conversation_titles" not in st.session_state:
        st.session_state.conversation_titles = []  # A list to hold conversation titles
    if "conversation_memories" not in st.session_state:
        st.session_state.conversation_memories = []  # A list to hold conversation memories
    if "active_conversation" not in st.session_state:
        st.session_state.active_conversation = -1  # An index to track the active conversation
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    # Check if the FAISS index exists, if not create it
    if not os.path.exists(INDEX_DIRECTORY):
        create_embeddings_from_pdfs()

    # Function to start a new chat
    def start_new_chat():
        st.session_state.active_conversation = -1
        st.session_state.user_question = ""

    with st.sidebar:
        st.title("📚 Conversation History")
        
        #  Show conversation titles
        if st.session_state.conversation_titles:
            for i, title in enumerate(st.session_state.conversation_titles):
                if st.button(title, key=f"title_{i}"):
                    st.session_state.active_conversation = i
                    st.session_state.user_question = ""
        
        st.markdown("---")
        if st.button("🆕 New conversation"):
            start_new_chat()

    # Input for user question
    def submit_question():
        st.session_state.user_question = st.session_state.question_input
    
    user_question = st.text_input("❓ Enter your question:", key="question_input", 
                                 value=st.session_state.user_question,
                                 on_change=submit_question)

    # Display the active conversation title
    active_messages = []
    active_memory = None
    
    # If there is an active conversation, get the messages and memory
    if st.session_state.active_conversation >= 0:
        active_messages = st.session_state.conversations[st.session_state.active_conversation]
        active_memory = st.session_state.conversation_memories[st.session_state.active_conversation]
    
    # If the user has entered a question and it's not already in the active messages
    if st.session_state.user_question and (not active_messages or st.session_state.user_question not in [msg[1] for msg in active_messages if msg[0] == "🧑"]):
        with st.spinner("Yanıt hazırlanıyor..."):
            # Check if the active conversation is -1 (no conversation)
            if st.session_state.active_conversation == -1:
                new_title = f"Conversation {len(st.session_state.conversations)+1}"
                st.session_state.conversation_titles.append(new_title)
                st.session_state.conversations.append([])
                st.session_state.conversation_memories.append(ConversationBufferMemory(memory_key="chat_history", return_messages=True))
                st.session_state.active_conversation = len(st.session_state.conversations) - 1
                active_messages = st.session_state.conversations[st.session_state.active_conversation]
                active_memory = st.session_state.conversation_memories[st.session_state.active_conversation]
            
            # Get the chain and memory for the active conversation
            chain, _ = get_chain(active_memory)
            response = chain.invoke({"question": st.session_state.user_question})
            
            # Add the user question and response to the active messages
            active_messages.append(("🧑", st.session_state.user_question))
            active_messages.append(("🤖", response["answer"]))
            
            display_text_with_effect(response["answer"])
            
            # Update the conversation history
            st.session_state.conversations[st.session_state.active_conversation] = active_messages

    #  Display the conversation history
    if active_messages:
        st.markdown("---")
        for sender, message in active_messages:
            if sender == "🧑":
                st.markdown(f"**{sender} You:** {message}")
            else:
                st.markdown(f"**{sender} DEUbot:** {message}")

if __name__ == "__main__":
    main()