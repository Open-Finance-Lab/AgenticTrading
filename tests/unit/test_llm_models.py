#!/usr/bin/env python3
"""
Unit tests for LLM model integration
测试LLM模型集成的单元测试
"""

import os
import asyncio
import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json

# Load environment variables
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)


class TestLLMModels:
    """Test LLM model integration and functionality"""
    
    @pytest.fixture(scope="class")
    def llm_client(self):
        """Create LLM client for testing"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in environment")
        return AsyncOpenAI(api_key=api_key)
    
    @pytest.mark.asyncio
    async def test_o4_mini_basic_functionality(self, llm_client):
        """Test o4-mini model basic functionality"""
        try:
            response = await llm_client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello world"}
                ],
                max_completion_tokens=100
            )
            
            assert response.choices[0].message.content is not None
            assert len(response.choices[0].message.content.strip()) > 0
            
        except Exception as e:
            pytest.fail(f"o4-mini basic test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_o4_mini_trading_signal_json(self, llm_client):
        """Test o4-mini model for trading signal generation with JSON output"""
        try:
            response = await llm_client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": "You are a trading analyst. Always respond with valid JSON."},
                    {"role": "user", "content": '''
                        Analyze AAPL stock and provide a trading signal.
                        Respond only with JSON: {"signal": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reasoning": "brief explanation"}
                    '''}
                ],
                max_completion_tokens=300
            )
            
            # Check response exists
            assert response.choices[0].message.content is not None
            
            # Try to parse as JSON
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            
            # Validate required fields
            assert "signal" in parsed
            assert parsed["signal"] in ["BUY", "SELL", "HOLD"]
            assert "confidence" in parsed
            assert isinstance(parsed["confidence"], (int, float))
            assert 0.0 <= parsed["confidence"] <= 1.0
            assert "reasoning" in parsed
            assert isinstance(parsed["reasoning"], str)
            
        except json.JSONDecodeError as e:
            pytest.fail(f"JSON parsing failed: {e}. Response: {response.choices[0].message.content}")
        except Exception as e:
            pytest.fail(f"o4-mini trading signal test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_o4_mini_parameter_requirements(self, llm_client):
        """Test o4-mini model parameter requirements"""
        # Test that max_completion_tokens works (not max_tokens)
        try:
            response = await llm_client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "user", "content": "Brief test"}
                ],
                max_completion_tokens=50
            )
            assert response.choices[0].message.content is not None
        except Exception as e:
            pytest.fail(f"max_completion_tokens test failed: {e}")
        
        # Test that temperature parameter is not supported (should fail)
        with pytest.raises(Exception) as exc_info:
            await llm_client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "user", "content": "Brief test"}
                ],
                max_completion_tokens=50,
                temperature=0.1  # This should fail
            )
        assert "temperature" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_available_models(self, llm_client):
        """Test and log available models"""
        try:
            models = await llm_client.models.list()
            available_models = [model.id for model in models.data if 'gpt' in model.id or 'o4' in model.id]
            
            print(f"\n✅ Available GPT/o4 models: {available_models}")
            
            # Ensure o4-mini is available
            assert "o4-mini" in available_models, f"o4-mini not in available models: {available_models}"
            
        except Exception as e:
            pytest.fail(f"Model listing failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
