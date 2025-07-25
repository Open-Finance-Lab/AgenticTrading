#!/usr/bin/env python3
"""
测试端口冲突处理功能

这个脚本测试Alpha Agent Pool Core中的端口冲突检测和自动端口分配功能。
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from FinAgents.agent_pools.alpha_agent_pool.core import check_port_available, find_available_port

def test_port_functions():
    """测试端口检测和查找功能"""
    
    print("🧪 Testing port availability functions...")
    
    # Test port availability check
    print(f"Port 5051 available: {check_port_available(5051)}")
    print(f"Port 5052 available: {check_port_available(5052)}")
    print(f"Port 8080 available: {check_port_available(8080)}")
    
    # Test find available port
    print(f"Available port starting from 5051: {find_available_port(5051)}")
    print(f"Available port starting from 5052: {find_available_port(5052)}")
    
    # Test commonly used ports (should be occupied)
    common_ports = [22, 80, 443, 8000, 8001, 8002]
    for port in common_ports:
        available = check_port_available(port)
        print(f"Port {port}: {'available' if available else 'occupied'}")

if __name__ == "__main__":
    test_port_functions()
