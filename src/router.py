from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

#Loading the Model:
load_dotenv()
groq =os.getenv('groq_key')

#Setting the LLM:
llm=ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0,
    api_key=groq
)


#Defining the router Agent Based on query:
def router_query(query):

    #Setting the router prompt:
    router_prompt = ChatPromptTemplate.from_template("""
    You are a query router for a study companion.
    
    Classify the user's query into ONE of the following categories:
    
    - vectordb → conceptual, theory, explanations, definitions, AI/Machine Learning/Deep Learning topics, Large Language Processing, Statistical Understanding, Natural Language Processing, 
    - coding → programming, code generation, debugging, implementation
    - both → requires explanation + code
    
    Return ONLY the category name.
    
    Query:
    {query}
    """)

    output_parser=StrOutputParser()
    router= router_prompt| llm |output_parser

    #Running the query:
    results = router.invoke({"query": query})
    return results
