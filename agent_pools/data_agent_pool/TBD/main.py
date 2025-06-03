# main.py
import uvicorn
import agent_pools.data_agent_pool.TBD.app as app

if __name__ == "__main__":
    uvicorn.run(app.app, host="0.0.0.0", port=8001, reload=True)