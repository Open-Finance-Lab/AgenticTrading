"""
Custom Agent framework supporting OpenAI Function Calling for Alpha Research.
🧠 2025 Revision:
- Compatible with openai>=1.0 SDK
- Automatically detects tool function parameters (based on inspect.signature)
- Automatically passes context
"""

import os
import json
import inspect
from openai import OpenAI


# ==============================
# Tool function decorator
# ==============================
def function_tool(func, name=None, description=None):
    """Wrap a Python function as a callable tool"""
    func.is_tool = True
    func.name = name or func.__name__
    func.description = description or func.__doc__ or "No description available"
    return func


# ==============================
# Agent class definition
# ==============================
class Agent:
    """
    General-purpose agent class supporting OpenAI Function Calling with automatic tool execution.
    """

    def __init__(self, name="Agent", instructions="", model="gpt-4o-mini", tools=None):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _find_tool(self, name):
        """Find a tool by name in the registered tool list"""
        for t in self.tools:
            if t.__name__ == name or getattr(t, "name", None) == name:
                return t
        return None

    def _build_tool_schema(self, func):
        """Automatically generate JSON schema for function parameters"""
        sig = inspect.signature(func)
        params = {}
        required = []

        for name, param in sig.parameters.items():
            if name in ("ctx", "self"):
                continue

            # Infer parameter type
            ptype = "string"
            if param.annotation == int:
                ptype = "integer"
            elif param.annotation == float:
                ptype = "number"
            elif param.annotation == bool:
                ptype = "boolean"

            params[name] = {
                "type": ptype,
                "description": f"Argument {name}"
            }
            if param.default == inspect._empty:
                required.append(name)

        return {
            "type": "object",
            "properties": params,
            "required": required,
            "additionalProperties": True
        }

    def run(self, user_request, context=None, max_turns=10):
        """Core execution logic: GPT planning → automatic tool execution → result aggregation"""
        print(f"\n[Agent] Starting: {self.name}")
        print(f"[Agent] Model: {self.model}")
        print(f"[Agent] User request: {user_request[:200]}...")
        print(f"[Agent] Number of available tools: {len(self.tools)}")

        # Initial conversation context
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": user_request},
        ]

        for turn in range(max_turns):
            try:
                # Automatically build tool schemas
                tool_schemas = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.__name__,
                            "description": t.__doc__ or "No description provided",
                            "parameters": self._build_tool_schema(t)
                        }
                    }
                    for t in self.tools
                ]

                # Send request
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                )
            except Exception as e:
                print(f"OpenAI API call failed: {e}")
                return f"OpenAI API call failed: {e}"

            msg = response.choices[0].message

            # Check if a tool was called
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    name = call.function.name
                    args_str = call.function.arguments or "{}"

                    try:
                        args = json.loads(args_str)
                    except Exception:
                        args = {}

                    print(f"\n[Tool] Invoked: {name} | Args: {args}")

                    tool = self._find_tool(name)
                    if not tool:
                        print(f"[Warning] Tool {name} not registered.")
                        continue

                    try:
                        result = tool(context, **args) if context else tool(**args)
                        print(f"[Success] Tool executed: {name}")

                        # Feed the result back to the model
                        messages.append({
                            "role": "assistant",
                            "content": f"Tool {name} result: {str(result)[:1000]}"
                        })
                    except Exception as e:
                        print(f"[Error] Tool execution failed: {name} - {e}")
                        messages.append({
                            "role": "assistant",
                            "content": f"Error running {name}: {e}"
                        })

            else:
                # Model produced the final output
                final_output = msg.content or ""
                print("\n[Output] Final model response:\n", final_output[:800])
                return final_output

        return "Execution complete (maximum turns reached)"
