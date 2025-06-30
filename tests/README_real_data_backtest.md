# 使用真实数据的LLM增强回测

此测试展示了如何通过FinAgent orchestrator使用真实的AAPL和MSFT市场数据进行回测。

## 功能特性

- ✅ 通过Data Agent Pool MCP集成获取真实市场数据
- ✅ 动态LLM激活（仅在复杂市场条件下使用o4-mini）
- ✅ 智能回退到合成数据（当真实数据不可用时）
- ✅ 内存基础的归因分析
- ✅ 代理活动追踪
- ✅ 参数自适应调整

## 前置要求

1. **环境变量**：
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export POLYGON_API_KEY="your-polygon-api-key"
   ```

2. **依赖项**：
   ```bash
   pip install -r requirements.txt
   ```

3. **配置文件**：
   确保`config/polygon.yaml`文件存在并配置正确。

## 运行步骤

### 1. 启动Data Agent Pool服务器

在第一个终端中：
```bash
python start_data_agent_pool.py
```

等待看到：
```
🔧 正在启动MCP服务器在端口8001...
✅ DataAgentPoolMCPServer在0.0.0.0:8001启动成功
```

### 2. 运行回测测试

在第二个终端中：
```bash
python tests/test_simple_llm_backtest.py
```

或者作为测试运行：
```bash
python tests/test_simple_llm_backtest.py test
```

## 预期输出

测试将显示：

1. **初始化信息**：
   - 组件状态（orchestrator, LLM, memory agent）
   - 数据源配置
   - 真实数据加载状态

2. **实时进度**：
   - 交易日处理进度
   - 活跃代理
   - 信号源统计

3. **最终摘要**：
   - 性能指标（收益率、夏普比率、最大回撤）
   - 真实数据使用统计
   - LLM使用效率
   - 代理活动分析

## 数据源

- **真实数据**：通过Polygon.io API获取AAPL和MSFT的历史价格数据
- **合成数据**：当真实数据不可用时使用智能生成的回退数据
- **混合模式**：自动检测并混合使用真实和合成数据

## 架构

```
测试脚本 → MCP客户端 → Data Agent Pool → Polygon Agent → Polygon.io API
     ↓
   Orchestrator → Memory Agent → 归因分析
     ↓
   LLM Client → OpenAI o4-mini → 信号生成
```

## 故障排除

### 常见问题

1. **连接失败**：
   - 确保Data Agent Pool服务器正在运行
   - 检查端口8001是否可用

2. **API错误**：
   - 验证POLYGON_API_KEY和OPENAI_API_KEY
   - 检查API配额和限制

3. **内存错误**：
   - 确保内存代理正确初始化
   - 检查相关的依赖项

### 日志调试

增加日志详细程度：
```python
logging.basicConfig(level=logging.DEBUG)
```

## 性能指标

测试将报告：

- **总收益率**：整个回测期间的累计收益
- **年化收益率**：年化后的收益率
- **夏普比率**：风险调整后的收益指标
- **最大回撤**：历史最大资产缩水
- **波动率**：价格变动的标准差

## 扩展

要添加更多股票符号：

1. 修改测试中的symbols列表
2. 确保Data Agent Pool支持这些符号
3. 更新相关的配置文件

要使用不同的LLM模型：

1. 修改测试中的model参数
2. 调整相关的提示和参数
3. 测试模型响应格式的兼容性
