from pypdf import PdfReader
import warnings
from dotenv import find_dotenv, load_dotenv
import pickle
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
import os

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"

load_dotenv(find_dotenv())
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)


client = chromadb.PersistentClient(
    path="./chroma_db", settings=Settings(allow_reset=True)
)


def get_pdf_text(pdf_file):
    """
    Extract text from a PDF file.

    Parameters:
    - pdf_file (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF file.
    """
    pdf_page = PdfReader(pdf_file)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text


def create_docs(user_pdf_list):
    """
    Create Document objects from a list of PDF files.

    Parameters:
    - user_pdf_list (list): List of PDF file paths.

    Returns:
    list: List of Document objects.
    """
    docs = []
    for filename in user_pdf_list:

        chunks = get_pdf_text(filename)

        docs.append(
            Document(
                page_content=chunks,
                metadata={
                    "name": filename.name,
                    "id": filename.file_id,
                    "type=": filename.type,
                    "size": filename.size,
                },
            )
        )

    return docs


def create_embeddings():
    """
    Create SentenceTransformerEmbeddings object for creating document embeddings.

    Returns:
    SentenceTransformerEmbeddings: Instance of SentenceTransformerEmbeddings.
    """
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def push_to_db(splits, embeddings):
    """
    Push document splits and embeddings to the Chroma database.

    Parameters:
    - splits (list): List of Document objects.
    - embeddings (SentenceTransformerEmbeddings): Instance of SentenceTransformerEmbeddings.
    """
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        client=client,
        collection_name="my_collection",
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_similar_docs(query, k, flag):
    """
    Get similar documents based on a query using document embeddings.

    Parameters:
    - query (str): Query text.
    - k (int): Number of similar documents to retrieve.
    - flag (str): Flag for retrieval method ("Ensemble Retriever" or "similarity_search_with_score").

    Returns:
    list: List of similar documents.
    """
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db4 = Chroma(
        client=client,
        collection_name="my_collection",
        embedding_function=embeddings,
    )
    if flag == "Ensemble Retriever":
        with open("test.pickle", "rb") as fp:
            b = pickle.load(fp)
        retriever = db4.as_retriever(search_type="mmr")
        keyword_retriever = BM25Retriever.from_documents(b)
        keyword_retriever.k = k
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, keyword_retriever], weights=[0.5, 0.5]
        )
        docs = ensemble_retriever.invoke(query)

        return docs[0:k]
    else:
        return db4.similarity_search_with_score(query, k=k)


def delete_collection_db():
    """
    Reset the Chroma database by deleting the collection.

    Returns:
    None
    """
    client.reset()
    print("DB reset done")


def get_summary(current_doc):
    """
    Generate a summary for a given document.

    Parameters:
    - current_doc (Document): Input document for summarization.

    Returns:
    str: Generated summary for the input document.
    """
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary


def qa():
    """
    Initialize a ConversationalRetrievalChain for conversational question-answering.

    Returns:
    ConversationalRetrievalChain: Instance of ConversationalRetrievalChain.
    """
    return ConversationalRetrievalChain.from_llm(
        llm,
        db4.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        verbose=False,
    )


embeddings = create_embeddings()
db4 = Chroma(
    client=client,
    collection_name="my_collection",
    embedding_function=embeddings,
)
