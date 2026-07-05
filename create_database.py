# Standard library imports with sqlite3 override for Streamlit Cloud deployment compatibility
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Load PDF
loader = PyPDFLoader(
    r"C:\Users\salik\OneDrive\Desktop\Coding\RAG Project\document loaders\deeplearning.pdf"
)

docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)

# Create Mistral embeddings
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# Store embeddings in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db", 
)

print("Vector database created successfully!")