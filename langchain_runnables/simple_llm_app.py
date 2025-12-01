from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = PromptTemplate(
    template='Generate a 5 line report on {topic}',
    input_variables=['topic']
)

topic = input('Enter a topic: ')

formatted_prompt = prompt.format(topic=topic)

# FIX: use invoke instead of predict
response = llm.invoke([HumanMessage(content=formatted_prompt)])

print("Generated:\n", response.content)
