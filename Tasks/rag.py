from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

df = pd.read_csv("students.csv")

texts = []
for _, row in df.iterrows():
    texts.append(
        f"Student {row['Student_Name']} scored "
        f"Maths {row['Maths']}, "
        f"Physics {row['Physics']}, "
        f"Chemistry {row['Chemistry']}, "
        f"Computer Science {row['Computer_Science']}."
    )

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = splitter.create_documents(texts)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever()

def ask(query):
    docs = retriever.invoke(query)
    
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are a helpful assistant.
Answer the question based on the context below.

Context: {context}

Question: {query}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    print("🔥 Groq Chatbot + RAG Ready!")

    while True:
        q = input("You: ")
        if q.lower() == 'exit':
            break
        print("Bot:", ask(q))