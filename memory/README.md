# 🧠 Intelligent Memory System

### Abstract

This project implements an advanced, server-based memory system for Large Language Models (LLMs). It leverages a vector database (ChromaDB) for efficient information retrieval and exposes memory operations via a standardized server protocol. The system is designed to provide external LLM agents with a dynamic, queryable, long-term memory that overcomes standard context window limitations, enabling more accurate and context-aware responses.

### Project Status

The project is currently in a **prototyping and evaluation phase**. The core server and retrieval mechanisms are functional, and performance is being actively measured through dedicated latency and accuracy tests.

### Architecture

#### Directory Structure

```
/
├── testing/
│   ├── accuracy_testing.py   # Tests the model's ability to find facts
│   └── latency_test.py       # Measures performance with large contexts
├── chroma_retriever.py         # Manages interaction with ChromaDB vector store
├── memory_agent.py             # Client example of an OpenAI agent using the server
├── memory_server.py            # Core MCP server exposing memory tools
├── pydantic_models.py          # Pydantic schemas for data validation
├── server_testing.py           # Deprecated or alternative server implementation
├── test_client.py              # Simple client for unit testing server tools
├── sp500_headlines_2... .csv  # Sample dataset for testing
└── README.md                   # This documentation file
```

### Implementation Details

#### Core Components

* **Memory Server (`memory_server.py`)**
    The heart of the application, built using `FastMCP`. It exposes core memory operations as tools that external agents can call.

    * **Tool `store_memory`**: Allows an agent to save textual information along with structured metadata (category, source, timestamp) into the ChromaDB database.

    * **Tool `retrieve_memory`**: Enables an agent to query the memory system using a natural language string to find the most relevant stored information via semantic search.

* **Chroma Retriever (`chroma_retriever.py`)**
    This module is an abstraction layer over the ChromaDB vector database. It handles all data storage and retrieval logic.

    * **Embedding**: Automatically converts text documents into vector embeddings using a `SentenceTransformer` model (`all-MiniLM-L6-v2`).

    * **Storage**: Adds documents, their vector embeddings, and associated metadata to a specified ChromaDB collection.

    * **Semantic Search**: Executes vector similarity searches to find documents most relevant to a given query.

* **Agent Client (`memory_agent.py`)**
    A standalone client application demonstrating how an external AI agent (powered by OpenAI's `gpt-4o`) can interact with the Memory Server.

    * **Tool Definitions**: It is configured with JSON definitions for `store_memory` and `retrieve_memory`, enabling the LLM to decide when to call these functions.

    * **MCP Communication**: Uses the `mcp.client` library to communicate with the Memory Server, sending tool call requests and parsing the results.

    * **Autonomous Operation**: Implements a conversational loop where the LLM can make multiple tool calls in a row to achieve a goal.

* **Pydantic Models (`pydantic_models.py`)**
    Defines strict, type-safe data schemas for all API interactions and data structures. This includes models for tool inputs (`StoreMemoryInput`, `RetrieveMemoriesInput`), tool outputs, and the MCP protocol itself, ensuring data integrity and clear contracts.

### Testing Framework

The project includes a multi-faceted testing suite to ensure functionality and evaluate performance.

#### Unit & Integration Testing

* The `test_client.py` script serves as a simple integration test, making direct calls to the `store_memory` and `retrieve_memory` tools on a running `memory_server.py` instance to validate basic functionality.

#### Performance & Accuracy Evaluation

The `/testing` directory contains scripts for more advanced evaluation against real-world data (`sp500_headlines_2... .csv`).

* **`latency_test.py`**: Measures system performance by sending progressively larger chunks of text data to the model and plotting the latency and throughput. This tests the "Scalability" challenge.

* **`accuracy_testing.py`**: Tests the "Information Retrieval" challenge by providing the model a massive context (up to 120,000 tokens) and asking it to find a specific fact ("needle-in-a-haystack") buried within it.

### Getting Started

#### Prerequisites

* Python 3.9+

* An OpenAI API Key set as an environment variable: `OPENAI_API_KEY`.

* Required Python packages as listed in `requirements.txt`.

#### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies:**
    *`requirements.txt`*

    ```bash
    pip install openai pandas matplotlib chromadb sentence-transformers "mcp-client" "mcp-server"
    ```

#### Running the Application

1.  **Set your environment variable:**

    ```bash
    export OPENAI_API_KEY='your-secret-key-here'
    ```

2.  **Start the Memory Server:**
    Open a terminal and run the server. It will handle requests for storing and retrieving memories.

    ```bash
    python memory_server.py
    ```

3.  **Run the Agent Client:**
    Open a *second* terminal to run the interactive agent client. This agent will connect to the server you just started.

    ```bash
    python memory_agent.py
    ```

### Dependencies

* **openai**: For interacting with GPT models.

* **pandas**: For loading and processing the sample CSV data.

* **matplotlib**: For plotting performance graphs in `latency_test.py`.

* **chromadb**: The vector database used for memory storage.

* **sentence-transformers**: For generating text embeddings.

* **mcp-client** & **mcp-server**: For the core client-server communication protocol.

### License

This component is part of the FinAgent-Orchestration project and is licensed under the OpenMDW License. See the [LICENSE](../LICENSE) file in the project root directory for details.

### Contact Information

* **Issue Tracking:** [Link to your GitHub Issues Page]

* **Maintainer:** [Your Name or Email Address]
