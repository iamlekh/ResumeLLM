import warnings
import streamlit as st
from utils import *
from dotenv import find_dotenv, load_dotenv
import pickle

warnings.filterwarnings("ignore")

load_dotenv(find_dotenv())
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


def main():
    """
    Main function to handle PDF upload, processing, and database operations.

    Uploads PDF files, processes them to create embeddings, and stores them in a database.
    Allows deletion of embeddings from the database.
    """
    st.set_page_config(page_title="Dump PDF to Pinecone - Vector Store")
    st.title("Please upload your files...📁 ")

    pdfs = st.file_uploader(
        "Only PDF files allowed", type=["pdf"], accept_multiple_files=True
    )

    if st.button("load to db", type="primary"):

        if len(pdfs) != 0:
            with st.spinner("Wait for it..."):

                final_docs_list = create_docs(pdfs)

                st.write("*Resumes uploaded* :" + str(len(final_docs_list)))

                embeddings = create_embeddings()
                st.write(f"👉Creating embeddings instance done ")
                with open("test.pickle", "wb") as fp:  # Pickling
                    pickle.dump(final_docs_list, fp)
                push_to_db(final_docs_list, embeddings)
                # keyword_ret1 = keyword_ret(final_docs_list)

            st.success("Successfully pushed the embeddings to ChromaDB")

        else:
            st.error("Please upload some file.")

    if st.button("delete db"):
        st.write(f"👉 Embeddings from DB deleted.")
        delete_collection_db()


if __name__ == "__main__":
    main()
