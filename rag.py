from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

VECTOR_STORE_PATH = "faiss_index"

def get_rag_chain():
    """
    Creates and returns a RAG chain for answering questions.
    """
    # 1. Load the Local Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 2. Define the Prompt Template
    prompt_template = """
    You are "The Christite Assistant," a knowledgeable and respectful AI assistant for CHRIST (Deemed to be University).
    Your primary function is to provide accurate and concise answers based ONLY on the provided context from the university's official documents.
    Do not use any external knowledge. If the answer is not found in the context, clearly state "I'm sorry, that information is not available in the provided documents."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ACCURATE ANSWER BASED ON CONTEXT:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 3. Initialize the LLM
    # NEW, CORRECTED LINE
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

    # 4. Construct the RAG Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain