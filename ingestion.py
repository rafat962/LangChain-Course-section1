import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
# Fix: Import GoogleGenerativeAIEmbeddings instead of OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter 

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader('./mediumblog1.txt', autodetect_encoding=True)
    documents = loader.load()
    
    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    print("Creating chunks...")
    docs = text_splitter.split_documents(documents)
    
    print("Encoding chunks using Gemini...")
    # Setup Gemini Embeddings using your API key from the environment
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        output_dimensionality=1536
    )
    
    print("Creating index...")
    index_name = os.getenv("INDEX_NAME")
    PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    
    print("Done!")