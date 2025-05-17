from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import config

# Define the prompt template
prompt = PromptTemplate(
    template="""
You are an expert assistant. Answer the following question based strictly on the context provided.
Do not include phrases like "Based on the provided transcript" or "According to the context". 
Just answer directly.

context:
{context}


Question: 
{question}

Answer:
""",
    input_variables=["context", "question"]
)

def build_qa_chain_gemini(vector_store: FAISS):
    """
    Builds a retrieval-augmented generation (RAG) chain using Gemini and FAISS.
    This function returns a chain that can answer questions based on the given vector store.
    """
    try:
        # Initialize retriever for similarity-based search
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Initialize Gemini (Google Generative AI) for answering questions
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            temperature=0.2,
            google_api_key=config.GEMINI_API_KEY,
        )

        # Format documents for the prompt
        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        # Parallel processing setup
        parallel = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        # Define output parser
        parser = StrOutputParser()

        # Build the chain: retrieve -> format -> run prompt -> run Gemini -> parse result
        chain = parallel | prompt | llm | parser
        return chain

    except Exception as e:
        print(f"Error while building Gemini QA chain: {e}")
        raise
