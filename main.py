# Standard library imports with sqlite3 override for Streamlit Cloud deployment compatibility
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize embedding model
embedding_model = MistralAIEmbeddings(model="mistral-embed")

# Load existing Chroma vector database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model,
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
)

# Initialize LLM
llm = ChatMistralAI(model="mistral-small-2506")

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use only the provided context to answer the question. If the context does not contain the answer, respond with 'I don't know.'",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)

print("\n===== RAG System Created =====\n")
print("Press 0 to exit the program.")

while True:
    query = input("\nYou: ")

    if query == "0":
        print("Exiting the program.")
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    # Combine retrieved documents into a single context
    context = "\n\n".join(doc.page_content for doc in docs)

    # Format the prompt
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    # Get response from the LLM
    response = llm.invoke(final_prompt)

    # Print response
    print(f"\nAI: {response.content}")
