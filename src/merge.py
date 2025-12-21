from langchain_core.runnables import RunnableLambda, RunnableParallel
from rag_agent import rag_calling
from coding_agent import generate_code



def merge_both_results(vector_result, coding_result):
    merged_answer = f"""
    ###  Knowledge / Conceptual Explanation
    {vector_result.strip()}
    
    ---
    
    ### 💻 Code / Implementation
    {coding_result.strip()}
    """
    return merged_answer

# Wrap functions as runnables
rag_runnable = RunnableLambda(rag_calling)
coding_runnable = RunnableLambda(generate_code)

# Parallel execution + merge:
def merge_answer(question):
    both_chain = (
        RunnableParallel
            (
            vector_database = rag_runnable,
            coding = coding_runnable
        )
        | RunnableLambda(lambda results: merge_both_results(results['vector_database'], results['coding']))
    )
    merger_results = both_chain.invoke({"question": question})
    return merger_results
