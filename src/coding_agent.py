import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
google_flash=os.getenv('google_flash')
os.environ["GOOGLE_API_KEY"] = google_flash

#Pratical code Agent:
flash=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=google_flash,
    temperature=0.7,
    max_tokens=None,
)

def generate_code(question):
    prompt = PromptTemplate(
        template="""
        You are a god-level coding assistant. Assist with user queries and answer the question: {question} in Python coding.
        The coding may related to AI, Machine Learning, Deep Learning or any other real-world questions.
        """,
        input_variables=["question"],

    )
    output_parser = StrOutputParser()

    chain = prompt | flash | output_parser
    results = chain.invoke({"question": question})
    return results
