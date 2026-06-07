import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import ChatPromptTemplate,SystemMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings , GoogleGenerativeAIChat
from langchain_pinecone import PineconeVectorStore


print("Loading index...")

embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        output_dimensionality=1536
    )

llm = GoogleGenerativeAIChat(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
    max_output_tokens=2048
)

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings
)


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a helpful assistant that answers questions based on the following context: {context}"),
        HumanMessage(content="{question}")
    ]
)