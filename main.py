from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


load_dotenv()

def main():
    information = """Bernard Arnault is a French business magnate, 
    investor, and art collector. 
    He is the chairman and chief executive officer (CEO) of LVMH Moët Hennessy Louis Vuitton SE,
      the world's largest luxury goods company. Arnault has been ranked as one of the richest people 
      in the world, with a net worth that has fluctuated over the years but has often been in the tens
        of billions of dollars. He has been involved in various philanthropic efforts and has a 
        significant influence on the luxury fashion industry.  Arnault's business acumen and strategic
          vision have played a crucial role in the growth and success of LVMH, which owns a portfolio 
          of prestigious brands such as Louis Vuitton, Dior, and Moët & Chandon. He has been recognized
            for his contributions to the fashion industry and has received numerous awards and honors 
            throughout his career. Arnault's leadership style is often described as visionary and 
            innovative, and he continues to be a prominent figure in the global business landscape."""
    
    summary_template = """
    Given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables = ["information"],
        template =summary_template
    )

    #llm = ChatOpenAI(model="gpt-5", temperature=0.9)
    llm = ChatOllama(model="gemma3:270m", temperature=0.9)
    chain = summary_prompt_template | llm

    response = chain.invoke(input = {"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
