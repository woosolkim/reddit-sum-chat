# RedditRAG

A Python-based chatbot that uses Retrieval-Augmented Generation (RAG) to summarize and answer questions about Reddit threads.

## 🌟 Features

- **Dynamic Database Building**: Fetches top posts from a specified subreddit and builds a vector database using ChromaDB.
- **AI-Powered Summaries**: Automatically generates a summary for each new Reddit post using the Gemini API.
- **Interactive Chat**: Ask questions in a CLI environment and get context-aware answers based on the indexed Reddit data.
- **RAG-Powered**: Leverages Google's Gemini model with a RAG pipeline to provide accurate, source-based responses.
- **Configurable**: Easily change models, subreddits, and other parameters via `config.yaml`.

## 🛠️ Tech Stack

- **Language**: Python
- **Core Libraries**:
  - **Reddit API**: `praw`
  - **LLM**: `google-generativeai` (for Gemini)
  - **Vector Database**: `chromadb`
  - **Embeddings**: `sentence-transformers`
  - **Text Splitting**: `langchain`
- **Configuration**: `pyyaml`, `python-dotenv`
- **CLI**: `argparse`

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A Reddit account and API credentials (`CLIENT_ID`, `CLIENT_SECRET`)
- A Google AI API Key (`GOOGLE_API_KEY`)

### Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd reddit-api
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    _(A `requirements.txt` file should be created for this project)_

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file in the root directory. **Do not share this file.**

    ```env
    # Reddit API Credentials
    CLIENT_ID="your_client_id"
    CLIENT_SECRET="your_client_secret"
    REDDIT_USER_AGENT="A descriptive user agent, e.g., MyRAGBot v0.1 by u/your_username"

    # Google AI API Key
    GOOGLE_API_KEY="your_google_api_key"
    ```

5.  **Review Configuration:**
    Check the `config.yaml` file. You can change the default subreddit, models, and other parameters here.
    ```yaml
    reddit:
      default_subreddit: "korea"
      default_limit: 10
      # ...
    models:
      embedding: "jhgan/ko-sroberta-multitask"
      generative: "gemini-1.5-flash"
    # ...
    ```

## Usage

The application has two modes: `build` and `chat`.

### 1. Build the Database

First, you must build the vector database with data from Reddit. The script will fetch posts, generate summaries, and index them.

```bash
# Build using default settings from config.yaml
python main.py build

# Or specify a different subreddit and post limit
python main.py build --subreddit "programming" --limit 50

# To delete the old database and rebuild it from scratch
python main.py build --rebuild
```

### 2. Chat with the Data

Once the database is built, you can start a chat session.

```bash
python main.py chat
```

The application will launch an interactive prompt. Ask your questions, and type `exit` or `종료` to quit.

```
대화를 시작합니다. 챗봇에게 질문을 입력하세요.
대화를 종료하려면 '종료' 또는 'exit'를 입력하세요.

나 >
```
