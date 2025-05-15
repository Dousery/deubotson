import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time
import shutil
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyasına API anahtarınızı ekleyin.")

INDEX_DIRECTORY = "faiss_index"
PDF_DIRECTORY = "pdfs"

def create_embeddings_from_pdfs():
    loader = DirectoryLoader(PDF_DIRECTORY, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(INDEX_DIRECTORY)

def get_conversational_chain():
    prompt_template = """
    1. Sana kullanıcının sorusunu ve ilgili metin alıntılarını sağlayacağım.
    2. Görevin, yalnızca sağlanan metin alıntılarını kullanarak Dokuz Eylül Üniversitesi adına cevap vermektir.
    3. Yanıtı oluştururken şu kurallara dikkat et:
    - Sağlanan metin alıntısında açıkça yer alan bilgileri kullan.
    - Metin alıntısında açıkça bulunmayan cevapları tahmin etmeye veya uydurmaya çalışma.
    4. Yanıtı, Türkçe dilinde ve anlaşılır bir şekilde ver.
    5. Kullanıcıya her zaman yardımcı olmaya çalış, ancak mevcut bilgilere dayanmayan yanıtlardan kaçın.
    6. Eğer \"Sen kimsin\" diye bir soru gelirse \"Ben Dokuz Eylül Üniversitesinin asistan botuyum, amacım sizlere üniversite hakkında bilgi sağlamak! Size nasıl yardımcı olabilirim?\" diye cevap ver.
    Eğer hazırsan, sana kullanıcının sorusunu ve ilgili metin alıntısını sağlıyorum.
    Context: \n {context}?\n
    Kullanıcı Sorusu: \n{question}\n
    Yanıt:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def display_text_with_effect(text):
    placeholder = st.empty()
    displayed_text = ""
    for token in text.split():
        displayed_text += token + " "
        placeholder.markdown(f"<div style='font-size:15px;'>{displayed_text}</div>", unsafe_allow_html=True)
        time.sleep(0.1)

def user_input(input):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local(INDEX_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(input)
        if not docs:
            st.warning("Sorunuzla ilgili kaynak bulunamadı. Lütfen farklı bir soru sorun.")
            return
    except Exception as e:
        st.error(f"FAISS veritabanı yüklenirken hata oluştu: {str(e)}")
        return

    chain = get_conversational_chain()
    try:
        response = chain({"input_documents": docs, "question": input}, return_only_outputs=True)
        display_text_with_effect(response["output_text"])
    except Exception as e:
        st.error(f"Yanıt üretilirken bir hata oluştu: {str(e)}")

def main():
    st.set_page_config(page_title="DEUbot", page_icon="🎓", layout="centered")

    st.header("Merhaba ben DEUbot 🤖")
    st.subheader("Üniversitemiz hakkında ne öğrenmek istersin?")

    # FAISS veritabanını temizleyip yeniden oluştur
    if os.path.exists(INDEX_DIRECTORY):
        shutil.rmtree(INDEX_DIRECTORY)
    create_embeddings_from_pdfs()

    with st.sidebar:
        st.title("DEUbot Ayarları")
        st.info("Bu chatbot, Dokuz Eylül Üniversitesi hakkında sorularınızı yanıtlamak için tasarlanmıştır.")
        st.success("✅ FAISS veritabanı yeniden oluşturuldu.")
        st.markdown("---")
        st.caption("© 2025 Dokuz Eylül Üniversitesi")

    user_question = st.text_input("Sorunuzu Giriniz:")
    if user_question:
        with st.spinner("Sorunuz işleniyor, lütfen bekleyin..."):
            user_input(user_question)
    else:
        st.info("Üniversite hakkında merak ettiğiniz konuları sorabilirsiniz.")

if __name__ == "__main__":
    main()
