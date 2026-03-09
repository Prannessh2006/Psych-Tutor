import os
import streamlit as st
import nltk
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from langchain_groq import ChatGroq
import time


st.set_page_config(
    page_title="PsychTutor",
    page_icon="🧠",
    layout="wide"
)


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

.sidebar .sidebar-content {
    background-color:#0f172a;
}

</style>
""", unsafe_allow_html=True)

st.title("🧠 PsychTutor")
st.caption("Your AI Psychology Learning Companion")



with st.sidebar:

    st.title("📚 Knowledge Base")

    if st.sidebar.button("Clear Chat"):
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

    st.markdown("---")

    st.info("Answers include citations like **[1]** showing the source text.")


BOOK_FILES = {
    "MIT Psychology": "./psychology_books/mit_psychology.pdf",
    "OpenStax Psychology": "./psychology_books/psychology_2e.pdf"
}

VECTOR_DB_PATH = "./vector_database"
EMBED_MODEL = "all-MiniLM-L6-v2"

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


GROQ_KEY = st.secrets["GROQ_API_KEY"]

LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_KEY
)


def load_books():

    paths = {}

    for name, path in BOOK_FILES.items():

        if not os.path.exists(path):
            raise Exception(f"{path} not found")

        paths[name] = path

    return paths


def extract_text(pdf):

    pages = []

    with pdfplumber.open(pdf) as doc:

        for p in doc.pages:

            text = p.extract_text()

            if text:
                pages.append(text)

    return pages

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

@st.cache_resource
def build_db():

    embed = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    collection = client.get_or_create_collection("psychology")

    if collection.count() > 0:
        return embed, collection

    paths = load_books()

    docs = []
    meta = []

    for book, path in paths.items():

        pages = extract_text(path)

        for i, page in enumerate(pages):

            cleaned = preprocess(page)

            chunks = chunk_text(cleaned)

            for chunk in chunks:

                docs.append(chunk)

                meta.append({
                    "book": book,
                    "page": i
                })

    embeddings = embed.encode(docs)

    ids = [str(i) for i in range(len(docs))]

    collection.add(
        documents=docs,
        embeddings=embeddings.tolist(),
        metadatas=meta,
        ids=ids
    )

    return embed, collection


embed, collection = build_db()


def search(query):

    q = preprocess(query)

    emb = embed.encode([q])[0]

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
- Do NOT mention any external websites or books.
- Do NOT invent sources.
- Only cite sources using numbers like [1], [2].
- If the answer is not present in the sources, say:
  "The provided textbooks do not contain enough information to answer this question."

Sources:
{context}

Question:
{query}

Provide a clear educational explanation.
"""

    response = LLM.invoke(prompt)

    answer = response.content.replace("**", "")

    return answer, results




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
                    unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
