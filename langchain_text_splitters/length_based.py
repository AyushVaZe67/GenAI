from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# text = """
# Aspiring data scientist with a strong foundation in computer engineering and 
# hands-on experience in machine learning and deep learning. Recently completed 
# training in ML, DL, and GenAI, and currently deepening my expertise in data science. 
# Enthusiastic about computer vision, with a keen interest in OpenCV and YOLO for image 
# and video-based applications. Also experienced in developing intuitive Android and 
# cross-platform apps with integrated AI features. Passionate about turning data into 
# practical, impactful solutions.
# """

# splitter = CharacterTextSplitter(
#     chunk_size=200,
#     chunk_overlap=0,
#     separator=''
# )

# result = splitter.split_text(text)

# print(result)

loader = PyPDFLoader('EndsemQAI.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[1].page_content)