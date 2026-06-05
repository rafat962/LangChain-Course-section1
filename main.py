from typing import List

from pydantic import BaseModel,Field

from dotenv import load_dotenv
import os
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient

from langchain_tavily import TavilySearch

# tavily = TavilyClient()

# @tool("search")
# def search(query: str) -> str:
#     """
#     Tool that performs a search and returns results. In a real implementation, this would call an external search API.
#     Args:
#         query (str): The search query.
#     Returns:
#         str: The search results.
#     """
#     print(f"Performing search for query: {query}")
#     return tavily.search(query)


class Source(BaseModel):
    """Schema for a source by the agent."""
    url: str = Field(description="The URL of the source.")

class AgentResponse(BaseModel):
    """Schema for the agent's response."""
    answer: str = Field(description="The answer to the user's query.")
    sources: List[Source] = Field(default=[], description="A list of sources used to generate the answer.")

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# llm = ChatOllama(model="gemma3:1b-it-qat", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
agent = create_agent(
    llm,
    tools=[TavilySearch()],
    system_prompt=SystemMessage(
        content="You are a helpful assistant that can perform searches."
    ),
    response_format=AgentResponse
)


def main():
    print("Hello from course-resourses!")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Search for 3 job posting for an ai engineer using langchain in the bay area on linkedin")]}
    )
    print(f"Agent response: {result}")


if __name__ == "__main__":
    main()