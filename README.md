# AI-Powered YouTube Summarizer and QA Tool with RAG and FAISS

## Overview
This project is an AI-powered tool designed to extract, summarize, and answer questions based on YouTube video transcripts. By leveraging advanced technologies like LangChain, FAISS, and Retrieval-Augmented Generation (RAG), the tool provides concise summaries and precise answers to user queries. The system is built with a user-friendly interface using Gradio, making it accessible to a wide range of users.

## Key Features
- **Video Transcript Extraction**: Automatically fetches transcripts from YouTube videos.
- **Summarization**: Generates concise summaries of video content.
- **Question Answering (QA)**: Answers specific user queries based on the video transcript.
- **RAG Architecture**: Combines retrieval-based methods with generative AI for accurate and context-aware responses.
- **FAISS Integration**: Utilizes Facebook AI Similarity Search (FAISS) for efficient vector storage and similarity search.
- **User-Friendly Interface**: Built with Gradio for an interactive and intuitive user experience.

## How It Works
### 1. Transcript Extraction
The tool uses the `youtube-transcript-api` to fetch transcripts from YouTube videos. It supports both manually created and auto-generated transcripts, prioritizing the former for better accuracy.

### 2. Text Processing and Chunking
The fetched transcript is processed and split into manageable chunks using the `RecursiveCharacterTextSplitter` from LangChain. This ensures that the text is optimized for embedding and retrieval.

### 3. Embedding and Vectorization
The processed text chunks are converted into embeddings using the `GoogleGenerativeAIEmbeddings` model. These embeddings are stored in a FAISS index, enabling efficient similarity search.

### 4. Retrieval-Augmented Generation (RAG)
The system employs a RAG architecture to combine retrieval-based methods with generative AI. When a user asks a question:
- Relevant transcript chunks are retrieved from the FAISS index based on the query.
- The retrieved context is passed to a language model (LLM) to generate a context-aware response.

### 5. Summarization
The tool uses a predefined prompt template and an LLM to generate concise summaries of the video content. This makes it easier for users to grasp the main points of lengthy videos.

## Technologies Used
- **LangChain**: For building LLM-powered applications.
- **FAISS**: For vector storage and similarity search.
- **Gradio**: For creating an interactive user interface.
- **Google Generative AI**: For embeddings and language model capabilities.
- **YouTube Transcript API**: For fetching video transcripts.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/yt_summarizer.git
   cd yt_summarizer
   ```
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv my_env
   source my_env/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Launch the application:
   ```bash
   python ytbot_gemini.py
   ```
2. Open the Gradio interface in your browser.
3. Enter the YouTube video URL and choose to either summarize the video or ask a question about it.

## Example Workflow
1. **Summarization**:
   - Input: YouTube video URL.
   - Output: A concise summary of the video content.
2. **Question Answering**:
   - Input: YouTube video URL and a specific question.
   - Output: A detailed answer based on the video transcript.

### Sample Output
![Sample Output](img/sample_output.png)

## Architecture Diagram
```plaintext
YouTube Video --> Transcript Extraction --> Text Processing --> Embedding --> FAISS Index --> RAG --> Summary/Answer
```

## Benefits
- Saves time by automating transcript analysis.
- Provides accurate and context-aware answers to user queries.
- Makes video content more accessible and insightful.

## Future Enhancements
- Support for multilingual transcripts and queries.
- Integration with other video platforms.
- Advanced analytics and visualization for video content.
