import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings , GoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
print("Loading index...")

embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        output_dimensionality=1536
    )

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
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
        ("system", "You are a helpful assistant that answers questions based on the following context: {context}"),
        ("human", "{question}"),
    ]
)
def format_docs(docs):
    """Formats the retrieved documents into a string for the prompt."""
    return "\n\n".join([doc.page_content for doc in docs])



# ==================================================================
# Implementation without LCEL
# ==================================================================
def _get_relevant_documents_without_lcel( query ):
    """Retrieves relevant documents based on the query."""
    #step 1: retrieve relevant documents
    docs = retriever.invoke(query)
    #step 2: return the retrieved documents
    context = format_docs(docs)
    # step 3: return the context for the prompt
    prompt = prompt_template.format(context=context, question=query)
    # step 4: Invoke the LLM with the prompt and return the answer
    answer = llm.invoke(prompt)
    return  answer

# ==================================================================
# Implementation with LCEL
# ==================================================================


from operator import itemgetter

def create_retrieval_chain():
    """Creates a retrieval chain using LCEL."""
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain

if __name__ == '__main__':
    question = "What is Pinecone in machine learning?"
    print("Retrieving relevant documents...")
    # answer = _get_relevant_documents_without_lcel(question)
    retrieval_chain = create_retrieval_chain()
    answer = retrieval_chain.invoke({"question": question})
    print("Answer:")
    print(answer)


