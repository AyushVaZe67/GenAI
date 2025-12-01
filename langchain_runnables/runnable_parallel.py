from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model='llama-3.1-8b-instant')

prompt = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a linkedIN post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt, llm, parser),
    'linkedin' : RunnableSequence(prompt2, llm, parser)
})

print(parallel_chain.invoke({'topic':'AI'}))

print('asd')