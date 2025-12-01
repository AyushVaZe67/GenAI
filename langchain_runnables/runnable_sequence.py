from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model='llama-3.1-8b-instant')

prompt = PromptTemplate(
    template='Generate a 1 line joke on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the below joke in 2 lines \n {text}',
    input_variables=['text']
)

parser2 = StrOutputParser()

parser = StrOutputParser()

chain = RunnableSequence(prompt,llm,parser,prompt2,llm,parser)

print(chain.invoke({'topic':'AI'}))

print('asd')