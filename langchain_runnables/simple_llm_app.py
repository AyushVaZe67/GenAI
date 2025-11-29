from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = PromptTemplate(
    template = 'Generate a 5 line report on {topic}',
    input_variable = ['topic']
)

topic = input('Enter a topic: ')

formatted_prompt = prompt.format(topic=topic)

report_title = llm.predict(formatted_prompt)

print('Genrated : ', report_title)
