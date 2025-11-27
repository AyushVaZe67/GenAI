from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

# FREE MODEL 1 → Groq LLaMA 3.1 (8B Instant)
model1 = ChatGroq(model="llama-3.1-8b-instant")

# FREE MODEL 2 → Google Gemini Flash 1.5
model2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Prompt to generate notes
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text:\n{text}',
    input_variables=['text']
)

# Prompt to generate quiz
prompt2 = PromptTemplate(
    template='Generate 5 short question-answers from the following text:\n{text}',
    input_variables=['text']
)

# Prompt to merge the results
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single clean document:\nNotes: {notes}\nQuiz: {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

# Run both models in parallel
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

# Merge results using free model
merge_chain = prompt3 | model1 | parser

# Full pipeline
chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are supervised learning methods used for classification, regression, and outlier detection.

Advantages:
- Effective in high dimensional spaces.
- Works even when number of dimensions > samples.
- Uses support vectors → memory efficient.
- Flexible with different kernel functions.

Disadvantages:
- Risk of overfitting when features >> samples.
- No direct probability estimates.
- Sparse input requires model trained on sparse data.
"""

result = chain.invoke({'text': text})
print(result)

# Optional: print chain graph
chain.get_graph().print_ascii()
