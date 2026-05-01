from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
  """Schema for a source used by the agent"""
  url:str = Field(description="The URL of the source") 

class AgentResponse(BaseModel):
  """Schema for agent response with answer and sources"""
  answer:str = Field(description="The agent's answer to the query")
  sources:List[Source] = Field(default_factory = List,
                                description="List of sources used to generate the answer")  

llm = ChatOpenAI(model="gpt-5", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke(
        {"messages": HumanMessage(content="Search for 3 AI Engineer roles using LangChain in Singapore on LinkedIn and list their details")},
    )
    print(result)


if __name__ == "__main__":
    main()
