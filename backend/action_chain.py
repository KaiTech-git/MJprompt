from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

INDEX_NAME = "mj-prompts"


def newFunction(x):
    return x * x


def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Define retrival
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    chat = ChatOpenAI(verbose=True, temperature=0)
    # Define the chain with specified llm and prompt template
    mj_chat_prompt = hub.pull("kaitech/midjourney-template2")
    stuff_document_chain = create_stuff_documents_chain(chat, mj_chat_prompt)

    # Perform retrival
    retrival_chain = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_document_chain
    )
    res = retrival_chain.invoke(input={"input": query})
    return res


if __name__ == "__main__":
    result = run_llm(
        query="photo of a orange British Shorthair cat plying on cat's playground in a flat near the window. A pine forest can be seen through the window."
    )
    print(result["answer"])
