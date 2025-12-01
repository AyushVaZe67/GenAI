from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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

parser = StrOutputParser()

joke_gen = RunnableSequence(prompt, llm, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explaination':RunnableSequence(prompt2, llm, parser)
})

final_chain = RunnableSequence(joke_gen, parallel_chain)

print(final_chain.invoke({'topic':'Gay'}))