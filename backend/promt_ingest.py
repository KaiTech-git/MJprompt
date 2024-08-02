from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_prompt():
    """Function load csv file containing proper MIDJOURNEY prompts to Pinecone. """
    loader = CSVLoader("Art Prompts without Subject.csv")
    # Rows in CSV will not be splitted to chunks as the longest prompt has 300 tokens size
    csv_raws = loader.load()
    print(f"Going to add {len(csv_raws)} to Pinecone")

    PineconeVectorStore.from_documents(csv_raws, embeddings, index_name="mj-prompts")

    print("***Loading to vectorstore done***")


if __name__ == "__main__":
    ingest_prompt()
