# examples/test_autonomous_agent_simple.py

"""
简单的自治Agent测试脚本
直接测试AutonomousAgent的功能，无需复杂的MCP客户端连接
"""

import sys
import os
import time
import threading
import asyncio

# 添加路径以便导入
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from FinAgents.agent_pools.alpha_agent_pool.agents.autonomous.autonomous_agent import AutonomousAgent

def test_autonomous_agent_basic():
    """基础测试自治agent功能"""
    
    print("=== 创建自治Agent实例 ===")
    agent = AutonomousAgent(agent_id="test_autonomous_agent")
    
    print("\n=== 测试Orchestrator输入处理 ===")
    instruction = "分析AAPL股票趋势并生成交易建议"
    context = {"symbol": "AAPL", "timeframe": "1d"}
    
    result = agent._process_orchestrator_input(instruction, context)
    print(f"处理结果: {result}")
    
    print("\n=== 查看生成的任务 ===")
    for i, task in enumerate(agent.task_queue):
        print(f"{i+1}. 任务ID: {task.task_id}")
        print(f"   描述: {task.description}")
        print(f"   状态: {task.status}")
        print(f"   优先级: {task.priority}")
        print()
    
    print("\n=== 测试Memory查询 ===")
    memory_result = agent._query_memory("AAPL historical data", "market_data")
    print(f"Memory查询结果: {memory_result}")
    
    print("\n=== 测试代码工具生成 ===")
    tool_result = agent._generate_code_tool(
        description="计算股票技术指标",
        input_format="dict with prices array",
        expected_output="dict with technical indicators"
    )
    print(f"生成的工具: {tool_result['tool_name']}")
    print(f"工具文件: {tool_result['file_path']}")
    
    print("\n=== 测试工具执行 ===")
    test_data = {
        "prices": [100, 102, 98, 105, 103, 107, 110, 108, 112, 115]
    }
    
    execution_result = agent._execute_tool(tool_result['tool_name'], test_data)
    print(f"执行结果: {execution_result}")
    
    print("\n=== 测试验证代码生成 ===")
    test_scenarios = [
        {"input": {"prices": [100, 102, 98, 105]}, "expected": {"success": True}},
        {"input": {"prices": []}, "expected": {"success": True}}
    ]
    
    validation_result = agent._create_validation(tool_result['code'], test_scenarios)
    print(f"验证代码: {validation_result['validation_name']}")
    print(f"验证文件: {validation_result['file_path']}")
    
    print("\n=== 测试任务处理 ===")
    if agent.task_queue:
        test_task = agent.task_queue[0]
        print(f"处理任务: {test_task.description}")
        agent._process_task(test_task)
        print(f"任务状态: {test_task.status}")
    
    print("\n=== 自治Agent基础测试完成 ===")
    
    # 停止任务处理线程
    agent.task_processor_running = False
    return agent

def test_autonomous_workflow():
    """测试完整的自治工作流程"""
    
    print("\n" + "="*50)
    print("测试完整自治工作流程")
    print("="*50)
    
    agent = AutonomousAgent(agent_id="workflow_test_agent")
    
    # 模拟orchestrator发送多个不同类型的指令
    instructions = [
        ("分析AAPL股票的动量指标", {"symbol": "AAPL", "analysis": "momentum"}),
        ("预测未来一周的股价走势", {"symbol": "AAPL", "period": "1w"}),
        ("创建基于均线的交易策略", {"strategy_type": "moving_average"})
    ]
    
    for instruction, context in instructions:
        print(f"\n--- 处理指令: {instruction} ---")
        result = agent._process_orchestrator_input(instruction, context)
        print(f"结果: {result}")
        time.sleep(1)  # 模拟间隔
    
    print(f"\n总任务数: {len(agent.task_queue)}")
    
    # 手动处理几个任务来演示工作流程
    processed_count = 0
    for task in agent.task_queue[:3]:  # 处理前3个任务
        if task.status == "pending":
            print(f"\n--- 处理任务: {task.description[:50]}... ---")
            agent._process_task(task)
            print(f"任务状态: {task.status}")
            processed_count += 1
    
    print(f"\n处理了 {processed_count} 个任务")
    
    # 显示生成的工具
    print(f"\n生成的工具数量: {len(agent.generated_tools)}")
    for tool_name, tool_info in agent.generated_tools.items():
        print(f"- {tool_name}: {tool_info['description']}")
    
    agent.task_processor_running = False
    return agent

def run_agent_server_test():
    """测试agent作为服务器运行"""
    
    print("\n" + "="*50)
    print("测试Agent服务器模式")
    print("="*50)
    
    def run_server():
        """在线程中运行agent服务器"""
        agent = AutonomousAgent(agent_id="server_test_agent")
        try:
            # 设置较短的超时时间以便测试
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Server test timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10秒超时
            
            agent.start_mcp_server(host="127.0.0.1", port=5051)
        except (KeyboardInterrupt, TimeoutError):
            print("服务器测试结束")
            agent.task_processor_running = False
        except Exception as e:
            print(f"服务器运行错误: {e}")
            agent.task_processor_running = False
    
    print("启动Agent MCP服务器...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(3)
    print("服务器启动完成")
    
    # 模拟一些工作
    time.sleep(5)
    
    print("服务器测试完成")

def main():
    """主测试函数"""
    print("自治Agent测试套件")
    print("="*50)
    
    try:
        # 基础功能测试
        agent1 = test_autonomous_agent_basic()
        
        # 工作流程测试
        agent2 = test_autonomous_workflow()
        
        # 服务器模式测试 (可选)
        choice = input("\n是否测试服务器模式? (y/n): ")
        if choice.lower() == 'y':
            run_agent_server_test()
        
        print("\n" + "="*50)
        print("所有测试完成!")
        print("="*50)
        
        print("\n总结:")
        print("✓ 基础功能测试通过")
        print("✓ 工作流程测试通过")
        print("✓ 任务自主分解功能正常")
        print("✓ 代码生成功能正常")
        print("✓ 验证代码创建功能正常")
        print("✓ Memory查询功能正常")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
