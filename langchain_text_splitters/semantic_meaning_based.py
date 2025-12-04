from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Your text
text = """
Aspiring data scientist with a strong foundation in computer engineering and 
hands-on experience in machine learning and deep learning. Recently completed 
training in ML, DL, and GenAI, and currently deepening my expertise in data science. 
Enthusiastic about computer vision, with a keen interest in OpenCV and YOLO for image 
and video-based applications. Also experienced in developing intuitive Android and 
cross-platform apps with integrated AI features. Passionate about turning data into 
practical, impactful solutions.
"""

# Initialize Groq embeddings (using OpenAI client with Groq endpoint)
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("GROQ_API_KEY"),  # Your Groq API key
    openai_api_base="https://api.groq.com/openai/v1",  # Groq endpoint
    model="text-embedding-3-small"  # Groq supports this model
)

# Or alternatively, create the chunker directly
semantic_chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"  # or "standard_deviation", "interquartile"
)

# Split the text
documents = semantic_chunker.create_documents([text])

# Print results
for i, doc in enumerate(documents):
    print(f"\n--- Semantic Chunk {i+1} ---")
    print(doc.page_content)
    print(f"Length: {len(doc.page_content)} characters")