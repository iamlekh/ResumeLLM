import streamlit as st
from dotenv import load_dotenv
from utils import *

import warnings

warnings.filterwarnings("ignore")


def main():
    """
    Main function for HR Resume Screening Assistance tool.

    Allows users to input job description, specify the number of resumes to return,
    and choose the retrieval method (Ensemble Retriever or Similarity Search with Score).
    Displays relevant resumes along with summaries based on the chosen method.

    Parameters:
    None

    Returns:
    None
    """
    load_dotenv()

    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance...💁 ")
    st.subheader("I can help you in resume screening process")

    job_description = st.text_area(
        "***Please paste the 'JOB DESCRIPTION' here...***", key="1"
    )
    document_count = st.text_input("***No.of 'RESUMES' to return***", key="2")
    per_flag = st.radio(
        "***Retriever ?...***",
        ["Ensemble Retriever", "Similarity Search with Score"],
        index=None,
    )
    submit = st.button("Help me with the analysis")

    if submit:
        with st.spinner("Wait for it..."):

            relavant_docs = get_similar_docs(
                job_description, int(document_count), per_flag
            )
            st.write(":heavy_minus_sign:" * 30)

            if per_flag == "Ensemble Retriever":

                for item in range(len(relavant_docs)):
                    st.subheader("👉 " + str(item + 1))

                    st.write("**File** : " + relavant_docs[item].metadata["name"])
                    with st.expander("Show me 👀"):
                        summary = get_summary(relavant_docs[item])
                        st.write("**Summary** : " + summary)
            else:
                relavant_docs = sorted(relavant_docs, key=lambda x: x[1], reverse=True)
                for item in range(len(relavant_docs)):
                    st.subheader("👉 " + str(item + 1))

                    st.write("**File** : " + relavant_docs[item][0].metadata["name"])
                    st.info("**Match Score** : " + str(relavant_docs[item][1]))
                    with st.expander("Show me 👀"):
                        summary = get_summary(relavant_docs[item][0])
                        st.write("**Summary** : " + summary)

        st.success("Hope I was able to save your time❤️")


if __name__ == "__main__":
    main()
