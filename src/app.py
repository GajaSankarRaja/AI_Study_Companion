#Importing the dependencies:
import streamlit as st
from router import router_query
from rag_agent import rag_calling
from coding_agent import generate_code
from merge import merge_answer

#Setting the page setup:
st.set_page_config(page_title="Study Companion", layout="centered")
st.title("RAG Companion for your AI Learning")


#Passinng to the router:
def rag_function(question):
    if question:
        with st.spinner('Loading RAG Model...'):
            router_result = router_query(question)
            if router_result=="vectordb":
                results = rag_calling(question)
            elif router_result=="coding":
                results = generate_code(question)
            else:
                results = merge_answer(question)

            st.write(results)

    else:
        st.write("Please enter a question!!!")

#Calling the Main-function:
if __name__=="__main__":
    # Prompting the question:
    question = st.text_input("Enter your question")
    rag_function(question)