# RAG_PDF
<!DOCTYPE html>
<html>
<head>
    <title>Conversational RAG with PDF and Chat History</title>
</head>
<body>

<h1>Conversational RAG with PDF and Chat History</h1>

<p>This project demonstrates a Retrieval-Augmented Generation (RAG) model integrated with PDF document handling and chat history management. The application allows users to upload PDF files and engage in a conversational interface that leverages both the content of the uploaded documents and the history of the ongoing chat to provide contextually relevant answers.</p>

<h2>Features</h2>
<ul>
    <li><strong>PDF Upload and Processing:</strong> Users can upload multiple PDF files, which are processed and split into manageable chunks.</li>
    <li><strong>Chat History Management:</strong> Maintains and utilizes the chat history to improve the relevance and accuracy of responses.</li>
    <li><strong>Contextual Question Reformulation:</strong> Reformulates user questions based on chat history to ensure accurate retrieval of relevant context.</li>
    <li><strong>Concise Answer Generation:</strong> Generates concise responses based on retrieved context, with a maximum of three sentences.</li>
</ul>

<h2>Technologies Used</h2>
<ul>
    <li><strong>Streamlit:</strong> Framework for creating interactive web applications.</li>
    <li><strong>LangChain:</strong> Provides components for handling chat history, document retrieval, and question answering.</li>
    <li><strong>FAISS:</strong> A library for efficient similarity search and clustering of dense vectors.</li>
    <li><strong>Hugging Face Embeddings:</strong> Utilizes pre-trained embeddings for transforming text into vectors.</li>
    <li><strong>Groq:</strong> Utilized for running language models with API access.</li>
    <li><strong>PyPDFLoader:</strong> Handles loading and processing of PDF documents.</li>
    <li><strong>Dotenv:</strong> Manages environment variables for API keys and other sensitive information.</li>
</ul>

<h2>How It Works</h2>
<ol>
    <li><strong>Upload PDFs:</strong> Users upload one or more PDF files, which are saved temporarily and processed to extract text content.</li>
    <li><strong>Document Splitting:</strong> The extracted text is split into chunks for efficient handling and retrieval.</li>
    <li><strong>Vector Storage:</strong> Chunks of text are embedded into vectors and stored in a FAISS vector store for quick retrieval.</li>
    <li><strong>Chat History Management:</strong> Chat history is maintained to provide context to the assistant's responses, improving accuracy and relevance.</li>
    <li><strong>Contextual Question Reformulation:</strong> User questions are reformulated based on chat history and context to ensure accurate retrieval of relevant information.</li>
    <li><strong>Answer Generation:</strong> The assistant generates concise responses based on the retrieved context, with a maximum length of three sentences.</li>
</ol>

<h2>Setup and Usage</h2>
<ol>
    <li><strong>Install Dependencies:</strong> Ensure you have a `requirements.txt` file in the project directory with the necessary Python packages listed. Install the dependencies using:
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li><strong>Configure Environment:</strong> Create a `.env` file in the project directory with your API keys:
        <pre><code>GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token</code></pre>
    </li>
    <li><strong>Run the Application:</strong> Start the Streamlit application with:
        <pre><code>streamlit run app.py</code></pre>
    </li>
</ol>

</body>
</html>
