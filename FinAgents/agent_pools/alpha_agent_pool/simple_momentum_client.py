import httpx
import logging

class SimpleMomentumClient:
    """
    Simple HTTP client for calling the momentum agent's /generate_signal endpoint.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:5051"):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    async def call_generate_signal(self, symbol: str, price_list: list):
        """
        Calls the /generate_signal endpoint with symbol and price_list.
        """
        url = f"{self.base_url}/generate_signal"
        payload = {
            "symbol": symbol,
            "price_list": price_list
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    return resp.json()
                else:
                    self.logger.warning(f"HTTP {resp.status_code} for {url}: {resp.text}")
                    return {"signal": "HOLD", "confidence": 0.0, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            self.logger.error(f"Error calling {url}: {e}")
            return {"signal": "HOLD", "confidence": 0.0, "error": str(e)}
