"""
Custom Agent framework supporting OpenAI Function Calling for Alpha Research.
🧠 2025 修正版:
- 兼容 openai>=1.0 SDK
- 自动识别工具函数参数（基于 inspect.signature）
- 自动传入 context
"""

import os
import json
import inspect
from openai import OpenAI


# ==============================
# 工具函数装饰器
# ==============================
def function_tool(func, name=None, description=None):
    """将Python函数封装为可调用工具"""
    func.is_tool = True
    func.name = name or func.__name__
    func.description = description or func.__doc__ or "No description available"
    return func


# ==============================
# Agent 类定义
# ==============================
class Agent:
    """
    通用智能体类：支持 OpenAI Function Calling 自动执行工具。
    """

    def __init__(self, name="Agent", instructions="", model="gpt-4o-mini", tools=None):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _find_tool(self, name):
        """在注册的工具中查找函数"""
        for t in self.tools:
            if t.__name__ == name or getattr(t, "name", None) == name:
                return t
        return None

    def _build_tool_schema(self, func):
        """自动生成函数参数 JSON Schema"""
        sig = inspect.signature(func)
        params = {}
        required = []

        for name, param in sig.parameters.items():
            if name in ("ctx", "self"):
                continue

            # 推断参数类型
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
        """核心执行逻辑：GPT规划 → 自动调用工具 → 汇总输出"""
        print(f"\n🤖 [Agent] 启动: {self.name}")
        print(f"[Agent] 模型: {self.model}")
        print(f"[Agent] 用户请求: {user_request[:200]}...")
        print(f"[Agent] 可用工具数量: {len(self.tools)}")

        # 初始对话上下文
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": user_request},
        ]

        for turn in range(max_turns):
            try:
                # 自动构建工具schema
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

                # 发送请求
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                )
            except Exception as e:
                print(f"❌ OpenAI API 调用失败: {e}")
                return f"❌ OpenAI API 调用失败: {e}"

            msg = response.choices[0].message

            # 检查是否调用了工具
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    name = call.function.name
                    args_str = call.function.arguments or "{}"

                    try:
                        args = json.loads(args_str)
                    except Exception:
                        args = {}

                    print(f"\n🧰 调用工具: {name} | 参数: {args}")

                    tool = self._find_tool(name)
                    if not tool:
                        print(f"⚠️ 工具 {name} 未注册。")
                        continue

                    try:
                        result = tool(context, **args) if context else tool(**args)
                        print(f"✅ 工具执行成功: {name}")

                        # 将结果反馈给模型
                        messages.append({
                            "role": "assistant",
                            "content": f"工具 {name} 执行结果: {str(result)[:1000]}"
                        })
                    except Exception as e:
                        print(f"❌ 工具执行失败: {name} - {e}")
                        messages.append({
                            "role": "assistant",
                            "content": f"Error running {name}: {e}"
                        })

            else:
                # 模型生成最终结果
                final_output = msg.content or ""
                print("\n🧾 模型最终输出：\n", final_output[:800])
                return final_output

        return "✅ 执行完成（达到最大轮数）"
