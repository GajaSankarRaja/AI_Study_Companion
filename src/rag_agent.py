import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough


#Getting the Flash-API Key:
load_dotenv()
google_flash=os.getenv("google_flash")
os.environ["GOOGLE_API_KEY"] = google_flash

#Pratical code Agent:
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=google_flash,
    temperature=0.7,
    max_tokens=None,
)


#Embedding model:
embeddings_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


#Getting the Collection:
client=chromadb.PersistentClient(r"D:\Study_Companion\vector_Store")
collection = client.get_or_create_collection(
    name="Study_Companion",
    metadata={"Description": "RAG Implementation for Studying Artificial Intelligence and Data Science"}
)



#Reteriver:
def rag_reterive( query, top_k, score_threshold):
    print("Query: {query}")
    query_embedding=embeddings_model.encode([query],show_progress_bar=True)[0]
    try:
        print("Query: {query}")
        print("*"*5,"Querying the Collection","*"*5)
        results=collection.query(
            [query_embedding.tolist()],
            n_results=top_k
        )
        print("Retrieved {top_k} documents".format(top_k=top_k))
        retrieved_docs=[]
        if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
        else:
            print("No documents found")
        print("Answer is Fetched!!!!!")
        return retrieved_docs

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []




#Adding LLM to for generation:
def rag_calling(question):
    rag_retriever_results=rag_reterive(question,5,0.1)
    context = ("\n\n").join([doc['content'] for doc in rag_retriever_results]) if rag_retriever_results else ""
    if not context:
        return "No relevent context found"
    rag_prompt = ChatPromptTemplate.from_template("""
    You are an expert mentor with deep mastery in Artificial Intelligence, Statistics, Machine Learning, Deep Learning, Natural Language Processing, and Generative AI. Your goal is to teach concepts clearly, accurately, and intuitively.Use the provided context to answer the user’s question in a structured, helpful, and concise way.If the context is insufficient, state that and ask for clarification instead of guessing. Always prioritize factual accuracy and clear explanations.

    Context:
    {context}

    Question:
    {question}
    Instructions:
        - Read the context, but do not just copy it.
        - If th e context answers the question, summarize and explain in your own words.
        - If the context is insufficient, politely say you need more information.
        - Always start with a direct answer, then give examples or explanation.
    """)
    output_parser=StrOutputParser()
    rag_chain = (
        RunnableParallel({"question":RunnablePassthrough(),"context":lambda _:context})|
        rag_prompt|
        llm|
        output_parser
    )
    response=rag_chain.invoke({"question":question})
    print("Returning the Results:")
    return response
