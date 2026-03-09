import os
import streamlit as st
import nltk
import pdfplumber
import chromadb
import re
import time

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq


# Disable chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Ensure directories exist
os.makedirs("vector_database", exist_ok=True)

st.set_page_config(
    page_title="PsychTutor",
    page_icon="🧠",
    layout="wide"
)

# ---------------- UI STYLE ----------------

st.markdown("""
<style>

[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 14px;
}

.source-box{
    background-color:#111827;
    padding:14px;
    border-radius:10px;
    border:1px solid #2e2e2e;
    margin-bottom:10px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

st.title("🧠 PsychTutor")
st.caption("Your AI Psychology Learning Companion")

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("📚 Knowledge Base")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
PsychTutor answers questions using two psychology textbooks.
""")

    st.markdown("### MIT Introduction to Psychology")
    st.markdown("""
Topics include:
- Perception
- Cognition
- Memory
- Emotion
- Learning
""")

    st.markdown("### OpenStax Psychology")
    st.markdown("""
Topics include:
- Behavioral psychology
- Development
- Mental health
- Social psychology
""")

    st.info("Answers include citations like **[1]** showing the source text.")

# ---------------- CONSTANTS ----------------

BOOK_FILES = {
    "MIT Psychology": "./psychology_books/mit_psychology.pdf",
    "OpenStax Psychology": "./psychology_books/psychology_2e.pdf"
}

VECTOR_DB_PATH = "./vector_database"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---------------- NLTK SAFE LOAD ----------------

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ---------------- GROQ LLM ----------------

try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Missing GROQ_API_KEY in Streamlit secrets")
    st.stop()

LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_KEY
)

# ---------------- TEXT PROCESSING ----------------

def preprocess(text):

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = nltk.word_tokenize(text.lower())

    tokens = [t for t in tokens if t not in stop_words]

    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)

def chunk_text(text, chunk_size=500, overlap=100):

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):

        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

# ---------------- PDF EXTRACTION ----------------

def extract_text(pdf):

    pages = []

    with pdfplumber.open(pdf) as doc:

        for p in doc.pages:

            text = p.extract_text()

            if text:
                pages.append(text)

    return pages

# ---------------- MODEL LOAD ----------------

@st.cache_resource
def load_model():

    model = SentenceTransformer(EMBED_MODEL)

    return model

# ---------------- VECTOR DATABASE ----------------

@st.cache_resource
def build_db():

    model = load_model()

    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    collection = client.get_or_create_collection("psychology")

    if collection.count() > 0:
        return model, collection

    docs = []
    metas = []

    for book, path in BOOK_FILES.items():

        if not os.path.exists(path):
            st.error(f"{path} not found")
            st.stop()

        pages = extract_text(path)

        for i, page in enumerate(pages):

            cleaned = preprocess(page)

            chunks = chunk_text(cleaned)

            for chunk in chunks:

                docs.append(chunk)

                metas.append({
                    "book": book,
                    "page": i
                })

    embeddings = model.encode(docs)

    ids = [str(i) for i in range(len(docs))]

    collection.add(
        documents=docs,
        embeddings=embeddings.tolist(),
        metadatas=metas,
        ids=ids
    )

    return model, collection

model, collection = build_db()

# ---------------- SEARCH ----------------

def search(query):

    q = preprocess(query)

    emb = model.encode([q])[0]

    results = collection.query(
        query_embeddings=[emb.tolist()],
        n_results=4
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    combined = []

    for i in range(len(docs)):

        combined.append({
            "content": docs[i],
            "book": metas[i]["book"],
            "page": metas[i]["page"]
        })

    return combined

# ---------------- GENERATE ANSWER ----------------

def generate_answer(query):

    results = search(query)

    context = ""

    for i, r in enumerate(results):

        context += f"""
[{i+1}] Source: {r['book']} (Page {r['page']})

{r['content']}
"""

    prompt = f"""
You are an AI psychology tutor.

You must answer ONLY using the provided sources.

Rules:
- Do NOT mention external websites
- Do NOT invent sources
- Cite sources using [1], [2]

Sources:
{context}

Question:
{query}

Provide a clear explanation.
"""

    response = LLM.invoke(prompt)

    answer = response.content.replace("**", "")

    return answer, results

# ---------------- CHAT SYSTEM ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask a psychology question")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, results = generate_answer(prompt)

            placeholder = st.empty()

            full_text = ""

            for word in answer.split():

                full_text += word + " "

                placeholder.markdown(full_text)

                time.sleep(0.02)

            st.markdown("### 📚 Sources")

            for i, r in enumerate(results):

                snippet = r["content"][:200] + "..."

                st.markdown(
                    f"""
                    <div class="source-box">
                    <b>[{i+1}] {r['book']} — Page {r['page']}</b>
                    <br><br>
                    {snippet}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
