
def generated_tool_154e29d1(data):
    """
    计算股票技术指标
    
    Input format: dict with prices array
    Expected output: dict with technical indicators
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import json
    
    try:
        # 数据处理逻辑
        if isinstance(data, dict):
            # 处理字典输入
            result = process_dict_data(data)
        elif isinstance(data, list):
            # 处理列表输入
            result = process_list_data(data)
        else:
            # 处理其他类型
            result = process_generic_data(data)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "tool_name": "generated_tool_154e29d1"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "tool_name": "generated_tool_154e29d1"
        }

def process_dict_data(data):
    """处理字典格式数据"""
    if "prices" in data:
        prices = data["prices"]
        if isinstance(prices, list) and len(prices) > 1:
            # 计算价格变化
            changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            return {
                "price_changes": changes,
                "avg_change": sum(changes) / len(changes) if changes else 0,
                "volatility": np.std(changes) if changes else 0
            }
    return {"processed": True, "data_type": "dict"}

def process_list_data(data):
    """处理列表格式数据"""
    if all(isinstance(x, (int, float)) for x in data):
        # 数值列表
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": min(data),
            "max": max(data),
            "count": len(data)
        }
    return {"processed": True, "data_type": "list", "length": len(data)}

def process_generic_data(data):
    """处理通用数据"""
    return {
        "data_type": type(data).__name__,
        "processed": True,
        "summary": str(data)[:100]
    }
