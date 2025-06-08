# 🤖 DEUbot – Dokuz Eylul University Information Assistant

![image](https://github.com/user-attachments/assets/5cfe0a15-38ed-46ec-811e-b690724f695e)

DEUbot is an AI-powered information assistant that answers questions about Dokuz Eylül University. Developed using Gemini and LangChain technologies, this application analyzes university-related documents and provides intelligent responses to user inquiries.


---
## 🚀 Live Demo
https://deubot.streamlit.app

---
### ⚙️ Features

- 📄 Information extraction from PDFs: Automatically retrieves information from university-related PDF documents

- 💬 Conversation memory: Remembers previous interactions to provide contextual responses

- 🔍 Semantic search: Finds the most relevant content based on the user's question

- 🌐 Language support: It supports multilanguages.

- 🤖 Customized answers: Provides responses tailored to the DEUbot identity with a constrained answering structure

### 📁 Project Structure

```bash
.
├── app.py                  # Streamlit-based main application
├── pdfs/                   # University documents (PDF format)
├── faiss_index/            # Vector database using FAISS
├── .env                    # API key (environment variable)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
---

## ⚙️ Installation

---

### 1.  Clone the Repository

```bash
git clone https://github.com/Dousery/deubotson.git
cd deubotson
```


### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Dependencies
Create a .env file with the following content:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Run the App Locally

```bash
streamlit run app.py
```


---

## 📺 Demo Video

🖥️ [I will add youtube video here](LINK)

