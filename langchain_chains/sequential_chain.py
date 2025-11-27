from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a detailed report on {topic}',
    input_variable = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a 5 pointer summary from the following text \n {text}',
    input_variables = ['text']
)

model = ChatGroq(model="llama-3.1-8b-instant")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

res = chain.invoke({'topic' : 'India'})

print(res)

chain.get_graph().print_ascii()