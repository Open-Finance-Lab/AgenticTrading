# 🧠 Memory Agent

## 📐 System Overview

The Memory Agent serves as a specialized component within a broader financial agent orchestration framework. Its primary function is to provide a robust, persistent, and queryable memory layer for all other agents in the system. This enables learning, adaptation, and contextually aware decision-making by allowing agents to store and retrieve information from past actions, market events, and operational logs.

The Memory Agent operates as a centralized service, interacting with other agents via standardized protocols. It leverages a hybrid approach to data storage and employs advanced retrieval techniques to ensure relevant information is available when needed.

## 🏗️ Core Architecture & Functionality

The Memory Agent's architecture is designed for efficient memory management and standardized access:

* **MCP Server (`server_testing.py`):**
    * This is the heart of the Memory Agent, built using `FastMCP`.
    * It exposes memory operations as tools accessible via the Model Context Protocol (MCP).
    * **Key Tools Implemented:**
        * `store_memory`: Allows agents to save textual information along with structured metadata (category, source, timestamp, custom key-value pairs) into the memory system.
        * `retrieve_memory`: Enables agents to query the memory system using natural language or specific keywords to find the most relevant stored information, returning a specified number of results.
    * Integrates a lifespan manager (`app_lifespan`) for resource initialization and cleanup, ensuring the underlying database retriever is properly managed.
    * The server is designed to be run as an ASGI application (e.g., with Uvicorn), exposing an MCP endpoint (typically `/mcp`).

* **Data Retrieval & Storage (`chroma_retriever.py`):**
    * This module provides an abstraction layer over the underlying vector database, currently ChromaDB.
    * It handles:
        * **Embedding Generation:** Converts textual content into vector embeddings for semantic search (using Sentence Transformers).
        * **Storage:** Adds documents, their embeddings, and associated metadata to the ChromaDB collection.
        * **Semantic Search:** Performs similarity searches based on query embeddings to find relevant memories.
    * The current implementation primarily uses an in-memory ChromaDB instance, meaning data is reset when the MCP server stops. For persistence, this component would need to be configured to use a persistent ChromaDB backend.

* **OpenAI-Powered Client & RAG (`openai_mcp_client.py`):**
    * While not part of the Memory Agent *server* itself, this client demonstrates how an external AI agent (powered by OpenAI's GPT models) can interact with the Memory Agent.
    * **Tool Utilization:** It's configured with definitions of the `store_memory` and `retrieve_memory` tools, allowing the OpenAI model to decide when to call these functions based on user interaction.
    * **MCP Communication:** Uses `mcp.client` libraries to communicate with the Memory Agent's MCP server, sending tool call requests and receiving results.
    * **Retrieval Augmented Generation (RAG):** Implements a basic RAG flow. For user queries that aren't direct storage commands, it first calls `retrieve_memory` to fetch relevant context. This context is then provided to the OpenAI model along with the original query to generate more informed and grounded responses.
    * **Interaction Model:** Supports iterative tool calls and uses a low temperature for OpenAI responses to promote factuality.

## 🔗 Communication Protocols Utilized

While the broader FinAgent Orchestration Framework might use several protocols, the Memory Agent specifically relies on:

| Protocol | Role within Memory Agent Context                                      |
| :------- | :-------------------------------------------------------------------- |
| `MCP`    | Governs all interactions with the Memory Agent's tools (`store_memory`, `retrieve_memory`) from other agents or clients. Facilitates standardized data exchange for memory operations. |
| `HTTP(S)`| Underlies the MCP communication when using `streamablehttp_client` and `FastMCP`'s streamable HTTP application. |

*(Integration with A2A or other protocols would occur at the FinAgent Orchestration level, where other agents use these protocols to communicate among themselves and then use MCP to interact with this Memory Agent).*

## 📁 Project Structure (Memory Agent Component)