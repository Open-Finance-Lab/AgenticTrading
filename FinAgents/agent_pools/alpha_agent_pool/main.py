import uvicorn
from .app import app

if __name__ == "__main__":
    uvicorn.run(
        "alpha_agent_pool.app:app",
        host="localhost",
        port=8001,
        reload=True
    ) 