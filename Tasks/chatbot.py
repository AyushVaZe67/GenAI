import pandas as pd
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
# from langchain.chains.retrieval_qa.base import RetrievalQA

from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

df = pd.read_csv("students.csv")

texts = []
for _, row in df.iterrows():
    text = (
        f"Student {row['Student_Name']} scored "
        f"Maths {row['Maths']}, "
        f"Physics {row['Physics']}, "
        f"Chemistry {row['Chemistry']}, "
        f"Computer Science {row['Computer_Science']}."
    )
    texts.append(text)

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = splitter.create_documents(texts)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

while True:
    query = input("Ask: ")
    if query.lower() == "exit":
        break
    answer = qa_chain.run(query)
    print("Answer:", answer, "\n")