# PDF to Vector Store and Resume Screening Application

This application is designed to handle PDF uploads, process them to create embeddings, and store them in a vector database. It also provides a tool for HR professionals to screen resumes based on job descriptions using advanced retrieval methods.

## Features

### PDF Upload and Processing
- **PDF Upload**: Users can upload multiple PDF files through a user-friendly interface.
- **Text Extraction**: Extracts text from uploaded PDF files using `PdfReader`.
- **Document Creation**: Converts extracted text into `Document` objects with metadata.

### Embedding and Database Operations
- **Embeddings Creation**: Utilizes `SentenceTransformerEmbeddings` to create embeddings for documents.
- **Database Storage**: Stores document embeddings in a Chroma vector database.
- **Database Management**: Allows deletion of embeddings from the database to manage storage.

### Resume Screening Assistance
- **Job Description Input**: Users can input a job description to find relevant resumes.
- **Retrieval Methods**: Offers two retrieval methods:
  - **Ensemble Retriever**: Combines multiple retrieval strategies for enhanced accuracy.
  - **Similarity Search with Score**: Ranks documents based on similarity scores.
- **Result Display**: Shows relevant resumes with summaries and match scores.

### Conversational QA Bot
- **Interactive Q&A**: Users can ask questions about their documents, and the bot provides answers using a conversational retrieval chain.
- **Chat History**: Maintains a history of user queries and responses for context.

## Usage

1. **PDF Upload**: Navigate to the "Dump PDF to Pinecone - Vector Store" page and upload your PDF files.
2. **Load to Database**: Click the "load to db" button to process and store the documents.
3. **Resume Screening**: Go to the "Resume Screening Assistance" page, input a job description, specify the number of resumes, and choose a retrieval method.
4. **Conversational QA**: Use the "Docs QA Bot using Langchain" to interactively query your documents.

## Technical Details

- **Environment Variables**: Uses environment variables for configuration, loaded via `dotenv`.
- **Streamlit Interface**: Provides a web-based interface using Streamlit for easy interaction.
- **Chroma Database**: Utilizes Chroma for efficient vector storage and retrieval.
- **Language Model**: Integrates with OpenAI's `gpt-3.5-turbo` for language processing tasks.

## Installation

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Set up environment variables in a `.env` file.
4. Run the application using `streamlit run app.py`.
