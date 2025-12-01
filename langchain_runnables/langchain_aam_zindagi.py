import random

class NakliLLM:
    def __init__(self):
        print('LLM created!')

    def predict(self, prompt):
        response_list = [
            'New Delhi is capital of India',
            'AI stands for Artificial Intelligence',
            'Ayush is Goodboy'
        ]

        return random.choice(response_list)
    
class NakliPromptTemplate:
  def __init__(self, template, input_variables):
    self.template = template
    self.input_variables = input_variables

  def format(self, input_dict):
    return self.template.format(**input_dict)


class NakliLLMChain:
  def __init__(self, llm, prompt):
    self.llm = llm
    self.prompt = prompt

  def run(self, input_dict):

    final_prompt = self.prompt.format(input_dict)
    result = self.llm.predict(final_prompt)

    return result['response']


template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

prompt = template.format({'length':'short','topic':'india'})

llm = NakliLLM()
print(llm.predict('What is Laptop?'))

template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

chain = NakliLLMChain(llm, template)