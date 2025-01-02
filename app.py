import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

load_dotenv()

INDEX_DIRECTORY = "faiss_index"

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

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def display_text_with_effect(text):
    """
    Metni token token (kelime kelime) ekrana yazdırma efekti.
    """
    placeholder = st.empty()  # Dinamik güncellemeler için bir placeholder
    displayed_text = ""  # Gösterilen metni oluşturmak için bir string
    for token in text.split():
        displayed_text += token + " "  # Token'ları birleştir
        placeholder.markdown(f"<div style='font-size:15px;'>{displayed_text}</div>", unsafe_allow_html=True)
        time.sleep(0.1)  # Her token arasında gecikme ekle

def user_input(input):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        new_db = FAISS.load_local(INDEX_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(input)
    except Exception as e:
        st.error(f"FAISS veritabanı yüklenirken hata oluştu: {str(e)}")
        return

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": input}, return_only_outputs=True
    )

    # Cevabı yazdırma efektiyle ekrana göster
    display_text_with_effect(response["output_text"])

def main():
    st.set_page_config(page_title="DEUbot", page_icon="", layout="centered")
    st.header("Merhaba ben DEUbot 🤖")
    st.subheader("Üniversitemiz hakkında ne öğrenmek istersin?")

    # Kullanıcıdan soru al
    user_question = st.text_input("Sorunuzu Giriniz:")
    if user_question:
        with st.spinner("Sorunuz işleniyor, lütfen bekleyin..."):
            user_input(user_question)

if __name__ == "__main__":
    main()