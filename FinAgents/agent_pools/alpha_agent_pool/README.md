# Alpha Agent Pool Memory System

This README provides instructions for setting up the Neo4j database, installing dependencies, running the memory server, and testing the Alpha Agent Memory Client in this directory.

---

## 1. Neo4j Setup

### **A. Install Neo4j**
- Download and install Neo4j Community Edition from: https://neo4j.com/download/
- Start the Neo4j server (default port: 7687 for Bolt, 7474 for HTTP).

### **B. Set Credentials**
- Default credentials (as used in this repo):
  - **Username:** `neo4j`
  - **Password:** `FinOrchestration`
- You can change these in the code/config if needed.

### **C. Create Database and Indexes**
- The server will automatically create required indexes on startup.
- No manual schema setup is needed.

---


## 2. Running the Memory Server

From the project root directory, run:
```sh
uvicorn FinAgents.memory.memory_server:app --reload --port 8010
```
- The server will be available at: http://127.0.0.1:8010
- FastAPI docs: http://127.0.0.1:8010/docs

---

## 3. Testing the Alpha Agent Memory Client

### **A. Run the Test Script**
From the project root, run:
```sh
python -m FinAgents.agent_pools.alpha_agent_pool.tests.test_alpha_memory_client
```
- This will attempt to store and retrieve a test event using the running memory server.

### **B. Run the Client Directly**
You can also run the client directly (if it has a main block):
```sh
python -m FinAgents.agent_pools.alpha_agent_pool.alpha_memory_client
```

---

## 4. Troubleshooting

- **ModuleNotFoundError: No module named 'FinAgents'**
  - Make sure you are running commands from the project root and using `-m` for module execution.
- **ModuleNotFoundError: No module named 'mcp'**
  - Ensure all dependencies are installed and your `PYTHONPATH` includes the project root.
- **httpx.ConnectError: All connection attempts failed**
  - Make sure the memory server is running and accessible at `http://127.0.0.1:8010`.
- **Neo4j connection errors**
  - Ensure Neo4j is running and credentials match those in the code.

---

## 6. Useful Links
- [Neo4j Download](https://neo4j.com/download/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

---

For further help, check the code comments or contact the maintainers. 
