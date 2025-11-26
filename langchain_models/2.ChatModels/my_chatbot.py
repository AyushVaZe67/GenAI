from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

template = """You are a helpful assistant.
Answer clearly.

Question: {query}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["query"]
)

def ask(query):
    full_prompt = prompt.format(query=query)
    response = llm.invoke([HumanMessage(content=full_prompt)])
    return response.content

if __name__ == "__main__":
    print("🔥 Groq Chatbot Ready!")

    while True:
        q = input("You: ")
        if q == 'exit':
            break
        print("Bot:", ask(q))
