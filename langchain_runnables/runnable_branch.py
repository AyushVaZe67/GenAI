from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv
import os


load_dotenv()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

llm = ChatGroq(model='llama-3.1-8b-instant')

parser = StrOutputParser()

report_gen_chain = prompt1 | llm | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | llm | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))