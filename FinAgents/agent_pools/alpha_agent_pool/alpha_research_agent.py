"""
Alpha Research Agent - 基于OpenAI Agents SDK构建
集成AlphaAnalysisToolkit和AlphaVisualizationToolkit
"""
import os
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from pydantic import BaseModel

# OpenAI Agents SDK imports (本地agents.py已重命名为local_agents.py)
from agents import Agent, Runner, function_tool, RunContextWrapper

# 导入工具包
from alpha_analysis_toolkit import AlphaAnalysisToolkit
from alpha_visualization_toolkit import AlphaVisualizationToolkit


# ============================================================================
# Pydantic模型定义 - 用于结构化输出
# ============================================================================

class FactorPerformance(BaseModel):
    """因子预期表现"""
    market_regime: str
    confidence_level: float
    expected_sharpe: float
    
    class Config:
        extra = "forbid"  # 禁止额外字段


class FactorProposal(BaseModel):
    """单个因子建议"""
    factor_name: str
    description: str
    formula: str
    justification: str
    expected_performance: FactorPerformance
    
    class Config:
        extra = "forbid"  # 禁止额外字段


class AlphaFactorResponse(BaseModel):
    """LLM结构化因子建议响应"""
    factor_proposals: List[FactorProposal]
    market_summary: str
    risk_assessment: str
    
    class Config:
        extra = "forbid"  # 禁止额外字段


# ============================================================================
# Context类 - 用于在工具间传递状态
# ============================================================================

class AlphaResearchContext:
    """Alpha研究上下文 - 存储分析状态和数据"""
    def __init__(self):
        self.current_asset: Optional[str] = None
        self.market_data: Optional[pd.DataFrame] = None
        self.analysis_results: Dict[str, Any] = {}
        self.visualizations: Dict[str, Any] = {}
        self.factor_proposals: List[Dict] = []
        self.iteration_count: int = 0
        
        # 执行日志
        self.execution_log: List[Dict[str, Any]] = []
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_function_call(self, function_name: str, args: Dict, result: str, 
                         execution_time: float = 0):
        """记录函数调用"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'function_name': function_name,
            'arguments': args,
            'result_preview': result[:500] if len(result) > 500 else result,
            'result_length': len(result),
            'execution_time': execution_time,
            'step_number': len(self.execution_log) + 1
        }
        self.execution_log.append(log_entry)
        
    def save_session_log(self, output_dir: str = "agent_logs"):
        """保存session日志到文件"""
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"session_{self.session_id}.json")
        
        log_data = {
            'session_id': self.session_id,
            'execution_log': self.execution_log,
            'summary': {
                'total_steps': len(self.execution_log),
                'current_asset': self.current_asset,
                'iteration_count': self.iteration_count,
                'has_data': self.market_data is not None,
                'analysis_results_count': len(self.analysis_results),
                'visualizations_count': len(self.visualizations),
                'factor_proposals_count': len(self.factor_proposals)
            }
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return log_file


# ============================================================================
# 辅助函数
# ============================================================================

def _calculate_factor_performance(factor_name: str, 
                                  data: pd.DataFrame,
                                  signals: Dict[str, pd.Series],
                                  risk_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    计算因子的预期表现指标
    
    Args:
        factor_name: 因子名称
        data: 市场数据
        signals: 交易信号字典
        risk_metrics: 风险指标
        
    Returns:
        包含预期表现的字典
    """
    import numpy as np
    
    result = {
        'justification': '',
        'market_regime': '未知',
        'confidence_level': 0.5,
        'expected_sharpe': 0.0,
        'ic_mean': 0.0,
        'ic_std': 0.0,
        'calculation_method': '基于历史模拟'
    }
    
    try:
        # 1. 确定市场环境
        volatility = risk_metrics.get('volatility', 0)
        mean_return = risk_metrics.get('mean_return', 0)
        
        if volatility > 0.25:
            market_regime = "高波动"
        elif volatility > 0.15:
            market_regime = "中波动"
        else:
            market_regime = "低波动"
            
        if mean_return > 0.1:
            market_regime += "上涨"
        elif mean_return < -0.05:
            market_regime += "下跌"
        else:
            market_regime += "震荡"
        
        result['market_regime'] = market_regime
        
        # 2. 尝试从signals中获取对应的因子信号
        factor_signal = None
        
        # 匹配因子名称
        for sig_name, sig_data in signals.items():
            if factor_name.lower() in sig_name.lower() or sig_name.lower() in factor_name.lower():
                factor_signal = sig_data
                break
        
        # 3. 如果找到信号，计算IC（信息系数）和预期Sharpe
        if factor_signal is not None and len(factor_signal) > 20:
            # 计算收益率
            if 'Returns' in data.columns:
                returns = data['Returns']
            elif 'Close' in data.columns:
                returns = data['Close'].pct_change()
            else:
                returns = None
            
            if returns is not None and len(returns) > 20:
                # 去除NaN
                valid_idx = ~(factor_signal.isna() | returns.isna())
                clean_signal = factor_signal[valid_idx]
                clean_returns = returns[valid_idx]
                
                if len(clean_signal) > 20:
                    # 计算滚动IC（信息系数 = 因子值与未来收益的相关性）
                    # 简化版本：计算因子信号与下一期收益的相关性
                    future_returns = clean_returns.shift(-1)
                    
                    # 计算分段IC
                    window_size = min(20, len(clean_signal) // 5)
                    ic_values = []
                    
                    for i in range(0, len(clean_signal) - window_size, window_size):
                        window_signal = clean_signal.iloc[i:i+window_size]
                        window_future_ret = future_returns.iloc[i:i+window_size]
                        
                        if len(window_signal) > 5 and not window_future_ret.isna().all():
                            ic = window_signal.corr(window_future_ret)
                            if not np.isnan(ic):
                                ic_values.append(ic)
                    
                    if ic_values:
                        ic_mean = np.mean(ic_values)
                        ic_std = np.std(ic_values)
                        
                        # 计算预期Sharpe：简化公式 Sharpe ≈ IC_mean * sqrt(N) / IC_std
                        # 其中N是年化交易次数的平方根
                        trading_days_per_year = 252
                        # 判断数据频率（避免使用.freq属性，它可能不存在）
                        try:
                            freq_str = str(data.index.freq) if hasattr(data.index, 'freq') else ''
                        except:
                            freq_str = ''
                        
                        if 'hourly' in freq_str.lower() or 'h' in freq_str.lower() or len(data) > 1000:
                            # 高频数据
                            sqrt_n = np.sqrt(trading_days_per_year * 6)  # 假设每天6小时交易
                        else:
                            sqrt_n = np.sqrt(trading_days_per_year)
                        
                        if ic_std > 0:
                            expected_sharpe = abs(ic_mean) * sqrt_n / ic_std
                        else:
                            expected_sharpe = abs(ic_mean) * sqrt_n
                        
                        # 限制在合理范围
                        expected_sharpe = min(max(expected_sharpe, -2.0), 3.0)
                        
                        result['ic_mean'] = float(ic_mean)
                        result['ic_std'] = float(ic_std)
                        result['expected_sharpe'] = float(expected_sharpe)
                        
                        # 计算信心水平（基于IC的稳定性）
                        if len(ic_values) >= 10:
                            # IC值的一致性
                            positive_ic_ratio = sum(1 for ic in ic_values if ic > 0) / len(ic_values)
                            confidence = 0.5 + 0.3 * abs(ic_mean) + 0.2 * positive_ic_ratio
                            result['confidence_level'] = min(confidence, 0.95)
                        else:
                            result['confidence_level'] = 0.5
                        
                        # 生成理由
                        result['justification'] = (
                            f"基于历史回测，该因子的平均IC为{ic_mean:.3f}（标准差{ic_std:.3f}），"
                            f"显示{'正向' if ic_mean > 0 else '负向'}预测能力。"
                            f"在{market_regime}环境下，预期年化Sharpe比率约为{expected_sharpe:.2f}。"
                        )
                        result['calculation_method'] = f"滚动IC法（窗口={window_size}，样本数={len(ic_values)}）"
                        
                        return result
        
        # 4. 如果无法计算，使用基于因子类型的经验估计
        result['justification'] = f"基于{factor_name}因子类型和当前{market_regime}环境的经验估计"
        result['calculation_method'] = "经验估计（无历史信号数据）"
        
        # 根据因子类型给出经验Sharpe
        if 'momentum' in factor_name.lower():
            if 'trend' in market_regime or '上涨' in market_regime:
                result['expected_sharpe'] = 1.2
                result['confidence_level'] = 0.7
            else:
                result['expected_sharpe'] = 0.6
                result['confidence_level'] = 0.5
        elif 'revers' in factor_name.lower() or 'mean' in factor_name.lower():
            if '震荡' in market_regime:
                result['expected_sharpe'] = 1.0
                result['confidence_level'] = 0.65
            else:
                result['expected_sharpe'] = 0.5
                result['confidence_level'] = 0.45
        elif 'volatility' in factor_name.lower() or 'vol' in factor_name.lower():
            if '高波动' in market_regime:
                result['expected_sharpe'] = 0.9
                result['confidence_level'] = 0.6
            else:
                result['expected_sharpe'] = 0.7
                result['confidence_level'] = 0.55
        elif 'trend' in factor_name.lower():
            if 'trend' in market_regime or '上涨' in market_regime or '下跌' in market_regime:
                result['expected_sharpe'] = 1.3
                result['confidence_level'] = 0.75
            else:
                result['expected_sharpe'] = 0.4
                result['confidence_level'] = 0.4
        else:
            # 默认估计
            result['expected_sharpe'] = 0.8
            result['confidence_level'] = 0.5
        
        result['justification'] += f"。当前市场环境为{market_regime}，该类型因子通常表现{'较好' if result['expected_sharpe'] > 0.8 else '一般'}。"
        
    except Exception as e:
        result['justification'] = f"因子评估遇到错误: {str(e)}，使用默认估计"
        result['expected_sharpe'] = 0.5
        result['confidence_level'] = 0.3
        result['calculation_method'] = "默认值（计算失败）"
    
    return result


# ============================================================================
# 工具函数 - 使用@function_tool装饰器
# ============================================================================

@function_tool
def load_and_analyze_data(ctx: RunContextWrapper[AlphaResearchContext], 
                          csv_path: str, 
                          qlib_format: bool = False) -> str:
    """
    加载并分析资产数据，计算技术指标和信号
    
    Args:
        csv_path: CSV文件路径或qlib数据目录
        qlib_format: 是否使用qlib格式
        
    Returns:
        分析摘要字符串
    """
    start_time = datetime.now()
    try:
        print(f"\n🔧 [STEP {len(ctx.context.execution_log)+1}] 调用: load_and_analyze_data")
        print(f"   参数: csv_path={csv_path}, qlib_format={qlib_format}")
        
        # 1. 加载数据
        data = AlphaAnalysisToolkit.load_asset_data(csv_path, data_format="csv")
        data = AlphaAnalysisToolkit.preprocess_data(data, qlib_format=qlib_format)
        
        # 2. 计算技术指标
        indicators = AlphaAnalysisToolkit.calculate_technical_indicators(data, qlib_format=qlib_format)
        
        # 3. 生成交易信号
        signals = AlphaAnalysisToolkit.generate_alpha_signals(data, qlib_format=qlib_format)
        
        # 4. 计算风险指标
        risk_metrics = AlphaAnalysisToolkit.calculate_risk_metrics(data, qlib_format=qlib_format)
        
        # 5. 存储到context
        ctx.context.current_asset = csv_path
        ctx.context.market_data = data
        ctx.context.analysis_results = {
            'technical_indicators': indicators,
            'signals': signals,
            'risk_metrics': risk_metrics,
            'data_summary': {
                'rows': len(data),
                'columns': list(data.columns),
                'date_range': f"{data.index.min()} to {data.index.max()}"
            }
        }
        
        # 6. 生成摘要
        summary = f"""
✅ 数据加载成功: {csv_path}
📊 数据行数: {len(data)}
📅 时间范围: {data.index.min()} 至 {data.index.max()}
📈 技术指标: {len(indicators)} 个
🎯 交易信号: {len(signals)} 个
⚠️ 风险指标: 年化波动率={risk_metrics.get('volatility', 0):.2%}, Sharpe比率={risk_metrics.get('sharpe_ratio', 0):.2f}
"""
        
        # 记录执行
        execution_time = (datetime.now() - start_time).total_seconds()
        ctx.context.log_function_call(
            'load_and_analyze_data',
            {'csv_path': csv_path, 'qlib_format': qlib_format},
            summary,
            execution_time
        )
        print(f"   ✅ 执行完成 (耗时: {execution_time:.2f}秒)")
        
        return summary
        
    except Exception as e:
        error_msg = f"❌ 数据加载失败: {str(e)}"
        execution_time = (datetime.now() - start_time).total_seconds()
        ctx.context.log_function_call(
            'load_and_analyze_data',
            {'csv_path': csv_path, 'qlib_format': qlib_format},
            error_msg,
            execution_time
        )
        print(f"   ❌ 执行失败 (耗时: {execution_time:.2f}秒)")
        return error_msg


@function_tool
def create_visualizations(ctx: RunContextWrapper[AlphaResearchContext],
                         chart_types: str = "performance,factor_analysis") -> str:
    """
    创建可视化图表
    
    Args:
        chart_types: 图表类型字符串，用逗号分隔，如 "performance,heatmap,regime"
        
    Returns:
        可视化创建结果
    """
    start_time = datetime.now()
    print(f"\n🔧 [STEP {len(ctx.context.execution_log)+1}] 调用: create_visualizations")
    print(f"   参数: chart_types={chart_types}")
    
    if ctx.context.market_data is None:
        error_msg = "❌ 请先加载数据 (使用load_and_analyze_data工具)"
        ctx.context.log_function_call('create_visualizations', {'chart_types': chart_types}, error_msg, 0)
        return error_msg
    
    # 解析图表类型
    if isinstance(chart_types, str):
        chart_types = [t.strip() for t in chart_types.split(',')]
    elif chart_types is None:
        chart_types = ["performance", "factor_analysis"]
    
    try:
        viz_toolkit = AlphaVisualizationToolkit(theme="alpha_dark")
        data = ctx.context.market_data
        visualizations = {}
        
        # 根据请求创建图表
        if "performance" in chart_types and 'Returns' in data.columns:
            fig = viz_toolkit.create_performance_attribution(data[['Returns']])
            visualizations['performance'] = fig
            
        if "heatmap" in chart_types and data.shape[1] > 2:
            fig = viz_toolkit.create_factor_heatmap(data)
            visualizations['heatmap'] = fig
            
        if "regime" in chart_types and 'Close' in data.columns:
            fig = viz_toolkit.create_regime_detection_chart(data['Close'])
            visualizations['regime'] = fig
            
        if "factor_analysis" in chart_types:
            signals = ctx.context.analysis_results.get('signals', {})
            if signals and len(signals) > 0:
                # 将signals添加到data中以便可视化
                for name, signal in signals.items():
                    if isinstance(signal, pd.Series) and len(signal) == len(data):
                        data[name] = signal
                
                factor_names = list(signals.keys())[:5]  # 最多5个因子
                fig = viz_toolkit.create_alpha_factor_analysis(data, factor_names)
                visualizations['factor_analysis'] = fig
        
        # 存储到context
        ctx.context.visualizations = visualizations
        
        # 导出图表
        if visualizations:
            charts_list = list(visualizations.values())
            export_summary = viz_toolkit.export_charts_for_multimodal_input(charts_list)
            
            result = f"""
✅ 创建了 {len(visualizations)} 个可视化图表
📁 导出文件:
  - PNG: {len(export_summary.get('png_files', []))} 个
  - HTML: {len(export_summary.get('html_files', []))} 个
  - JSON: {len(export_summary.get('json_files', []))} 个
📊 图表类型: {', '.join(visualizations.keys())}
"""
            execution_time = (datetime.now() - start_time).total_seconds()
            ctx.context.log_function_call(
                'create_visualizations',
                {'chart_types': chart_types},
                result,
                execution_time
            )
            print(f"   ✅ 执行完成 (耗时: {execution_time:.2f}秒)")
            return result
        else:
            result = "⚠️ 未创建任何图表"
            execution_time = (datetime.now() - start_time).total_seconds()
            ctx.context.log_function_call('create_visualizations', {'chart_types': chart_types}, result, execution_time)
            print(f"   ⚠️ 执行完成 (耗时: {execution_time:.2f}秒)")
            return result
            
    except Exception as e:
        error_msg = f"❌ 可视化创建失败: {str(e)}"
        execution_time = (datetime.now() - start_time).total_seconds()
        ctx.context.log_function_call('create_visualizations', {'chart_types': chart_types}, error_msg, execution_time)
        print(f"   ❌ 执行失败 (耗时: {execution_time:.2f}秒)")
        return error_msg


@function_tool
def propose_alpha_factors(ctx: RunContextWrapper[AlphaResearchContext],
                         user_input: str = "") -> str:
    """
    基于市场分析提出alpha因子建议
    
    Args:
        user_input: 用户的补充输入或特殊要求
        
    Returns:
        因子建议列表
    """
    start_time = datetime.now()
    print(f"\n🔧 [STEP {len(ctx.context.execution_log)+1}] 调用: propose_alpha_factors")
    print(f"   参数: user_input={user_input if user_input else '(empty)'}")
    
    if ctx.context.market_data is None:
        error_msg = "❌ 请先加载数据 (使用load_and_analyze_data工具)"
        ctx.context.log_function_call('propose_alpha_factors', {'user_input': user_input}, error_msg, 0)
        return error_msg
    
    try:
        # 使用工具包的因子建议功能
        data = ctx.context.market_data
        signals = ctx.context.analysis_results.get('signals', {})
        risk_metrics = ctx.context.analysis_results.get('risk_metrics', {})
        factors = AlphaAnalysisToolkit.propose_alpha_factors(data, qlib_format=False)
        
        # 构建结构化建议（计算真实的预期表现）
        proposals = []
        for i, factor in enumerate(factors, 1):
            # 计算因子的预期表现
            factor_performance = _calculate_factor_performance(
                factor, data, signals, risk_metrics
            )
            
            proposal = {
                'factor_name': factor,
                'description': f"基于市场数据分析建议的{factor}因子",
                'formula': f"Ref({factor}, -1) / Ref({factor}, -5) - 1",  # 示例公式
                'justification': factor_performance['justification'],
                'expected_performance': {
                    'market_regime': factor_performance['market_regime'],
                    'confidence_level': factor_performance['confidence_level'],
                    'expected_sharpe': factor_performance['expected_sharpe'],
                    'ic_mean': factor_performance.get('ic_mean', 0),
                    'ic_std': factor_performance.get('ic_std', 0),
                    'calculation_method': factor_performance['calculation_method']
                }
            }
            proposals.append(proposal)
        
        # 存储到context
        ctx.context.factor_proposals = proposals
        
        # 格式化输出
        result = f"\n{'='*60}\n"
        result += f"🎯 Alpha因子建议 (基于数据分析)\n"
        result += f"{'='*60}\n\n"
        
        if user_input:
            result += f"💡 用户需求: {user_input}\n\n"
        
        for i, proposal in enumerate(proposals, 1):
            result += f"{i}. **{proposal['factor_name']}**\n"
            result += f"   描述: {proposal['description']}\n"
            result += f"   公式: {proposal['formula']}\n"
            result += f"   理由: {proposal['justification']}\n"
            result += f"   预期表现: Sharpe={proposal['expected_performance']['expected_sharpe']:.2f}\n\n"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        ctx.context.log_function_call(
            'propose_alpha_factors',
            {'user_input': user_input},
            result,
            execution_time
        )
        print(f"   ✅ 执行完成 (耗时: {execution_time:.2f}秒, 生成{len(proposals)}个因子)")
        
        return result
        
    except Exception as e:
        error_msg = f"❌ 因子建议生成失败: {str(e)}"
        execution_time = (datetime.now() - start_time).total_seconds()
        ctx.context.log_function_call('propose_alpha_factors', {'user_input': user_input}, error_msg, execution_time)
        print(f"   ❌ 执行失败 (耗时: {execution_time:.2f}秒)")
        return error_msg


@function_tool
def analyze_backtest_results(ctx: RunContextWrapper[AlphaResearchContext],
                             total_return: float = 0.0,
                             sharpe_ratio: float = 0.0,
                             max_drawdown: float = 0.0,
                             win_rate: float = 0.0) -> str:
    """
    分析回测结果并提供改进建议
    
    Args:
        total_return: 总收益率
        sharpe_ratio: Sharpe比率
        max_drawdown: 最大回撤
        win_rate: 胜率
        
    Returns:
        分析结果和改进建议
    """
    start_time = datetime.now()
    print(f"\n🔧 [STEP {len(ctx.context.execution_log)+1}] 调用: analyze_backtest_results")
    print(f"   参数: return={total_return}, sharpe={sharpe_ratio}, dd={max_drawdown}, wr={win_rate}")
    
    try:
        analysis = "\n" + "="*60 + "\n"
        analysis += "📊 回测结果分析\n"
        analysis += "="*60 + "\n\n"
        
        analysis += f"📈 总收益率: {total_return:.2%}\n"
        analysis += f"📊 Sharpe比率: {sharpe_ratio:.2f}\n"
        analysis += f"📉 最大回撤: {max_drawdown:.2%}\n"
        analysis += f"🎯 胜率: {win_rate:.2%}\n\n"
        
        # 提供改进建议
        analysis += "💡 改进建议:\n"
        
        if sharpe_ratio < 1.0:
            analysis += "  - Sharpe比率偏低，考虑优化因子权重或风险控制\n"
        
        if abs(max_drawdown) > 0.2:
            analysis += "  - 最大回撤较大，建议加强止损机制\n"
        
        if win_rate < 0.5:
            analysis += "  - 胜率偏低，考虑调整入场信号阈值\n"
        
        if total_return < 0:
            analysis += "  - 策略表现为负，需要重新评估因子有效性\n"
        else:
            analysis += "  - 策略整体表现良好，继续监控实盘表现\n"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        ctx.context.log_function_call(
            'analyze_backtest_results',
            {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            },
            analysis,
            execution_time
        )
        print(f"   ✅ 执行完成 (耗时: {execution_time:.2f}秒)")
        
        return analysis
        
    except Exception as e:
        error_msg = f"❌ 回测分析失败: {str(e)}"
        execution_time = (datetime.now() - start_time).total_seconds()
        ctx.context.log_function_call('analyze_backtest_results', {}, error_msg, execution_time)
        print(f"   ❌ 执行失败 (耗时: {execution_time:.2f}秒)")
        return error_msg


@function_tool
def generate_iteration_report(ctx: RunContextWrapper[AlphaResearchContext],
                              iteration_number: int) -> str:
    """
    生成研究迭代报告
    
    Args:
        iteration_number: 迭代次数
        
    Returns:
        迭代报告
    """
    start_time = datetime.now()
    print(f"\n🔧 [STEP {len(ctx.context.execution_log)+1}] 调用: generate_iteration_report")
    print(f"   参数: iteration_number={iteration_number}")
    
    ctx.context.iteration_count = iteration_number
    
    report = "\n" + "="*60 + "\n"
    report += f"📋 研究迭代报告 #{iteration_number}\n"
    report += "="*60 + "\n\n"
    
    report += f"📁 当前资产: {ctx.context.current_asset or '未加载'}\n"
    report += f"📊 数据状态: {'已加载' if ctx.context.market_data is not None else '未加载'}\n"
    report += f"🔬 分析结果: {len(ctx.context.analysis_results)} 项\n"
    report += f"📈 可视化图表: {len(ctx.context.visualizations)} 个\n"
    report += f"🎯 因子建议: {len(ctx.context.factor_proposals)} 个\n\n"
    
    report += "✨ 研究质量评分: "
    quality_score = 0
    if ctx.context.market_data is not None:
        quality_score += 25
    if ctx.context.analysis_results:
        quality_score += 25
    if ctx.context.visualizations:
        quality_score += 25
    if ctx.context.factor_proposals:
        quality_score += 25
    
    report += f"{quality_score}/100\n\n"
    
    # 添加执行日志摘要
    report += f"🔍 执行步骤: {len(ctx.context.execution_log)} 个函数调用\n"
    for log in ctx.context.execution_log:
        report += f"   [{log['step_number']}] {log['function_name']} (耗时: {log['execution_time']:.2f}s)\n"
    
    execution_time = (datetime.now() - start_time).total_seconds()
    ctx.context.log_function_call(
        'generate_iteration_report',
        {'iteration_number': iteration_number},
        report,
        execution_time
    )
    print(f"   ✅ 执行完成 (耗时: {execution_time:.2f}秒)")
    
    # 保存日志文件
    log_file = ctx.context.save_session_log()
    report += f"\n📄 完整执行日志已保存至: {log_file}\n"
    print(f"   📄 日志文件: {log_file}")
    
    return report


# ============================================================================
# Alpha Research Agent类
# ============================================================================

class AlphaResearchAgent:
    """
    Alpha研究智能体 - 基于OpenAI Agents SDK
    负责量化交易策略开发和分析
    """
    
    def __init__(self):
        """初始化Agent"""
        # 创建context实例
        self.context = AlphaResearchContext()
        
        # 创建Agent实例
        self.agent = Agent(
            name="AlphaResearchAgent",
            instructions="""
你是一个专业的量化交易研究员，专注于alpha因子研究和策略开发。

你的主要职责:
1. 加载和分析市场数据
2. 计算技术指标和风险指标
3. 生成交易信号
4. 创建可视化图表
5. 提出alpha因子建议
6. 分析回测结果

工作流程:
1. 首先使用load_and_analyze_data加载数据
2. 然后使用create_visualizations创建图表
3. 使用propose_alpha_factors提出因子建议
4. 如果有回测结果，使用analyze_backtest_results分析
5. 最后使用generate_iteration_report生成报告

始终保持专业、详细和数据驱动的分析方式。
""",
            model="o4-mini",
            tools=[
                load_and_analyze_data,
                create_visualizations,
                propose_alpha_factors,
                analyze_backtest_results,
                generate_iteration_report
            ]
        )
    
    async def run_analysis(self, user_request: str) -> str:
        """
        运行分析任务
        
        Args:
            user_request: 用户请求
            
        Returns:
            分析结果
        """
        try:
            print(f"\n{'='*70}")
            print(f"🚀 开始Alpha研究分析")
            print(f"   Session ID: {self.context.session_id}")
            print(f"{'='*70}\n")
            
            result = await Runner.run(
                self.agent,
                user_request,
                context=self.context,
                max_turns=10
            )
            
            # 保存最终执行日志
            log_file = self.context.save_session_log()
            
            print(f"\n{'='*70}")
            print(f"✅ 分析完成")
            print(f"   总执行步骤: {len(self.context.execution_log)}")
            print(f"   日志文件: {log_file}")
            print(f"{'='*70}\n")
            
            return result.final_output
        except Exception as e:
            error_msg = f"❌ 分析失败: {str(e)}"
            # 即使失败也保存日志
            try:
                log_file = self.context.save_session_log()
                print(f"   错误日志已保存: {log_file}")
            except:
                pass
            return error_msg
    
    def run_analysis_sync(self, user_request: str) -> str:
        """
        同步运行分析任务
        
        Args:
            user_request: 用户请求
            
        Returns:
            分析结果
        """
        return asyncio.run(self.run_analysis(user_request))
    
    def run_complete_workflow(self, csv_path: str, user_input: str = "") -> str:
        """
        运行完整的分析工作流
        
        Args:
            csv_path: CSV文件路径
            user_input: 用户补充输入
            
        Returns:
            完整工作流结果
        """
        request = f"""
请对文件 {csv_path} 进行完整的alpha研究分析:

1. 加载并分析数据
2. 创建可视化图表（包括performance、factor_analysis、regime）
3. 提出alpha因子建议
4. 生成第1次迭代报告

用户需求: {user_input if user_input else '标准分析流程'}
"""
        return self.run_analysis_sync(request)
    
    def print_execution_log(self, detailed: bool = True):
        """
        打印执行日志
        
        Args:
            detailed: 是否显示详细信息
        """
        print("\n" + "="*70)
        print(f"📊 执行日志 - Session: {self.context.session_id}")
        print("="*70 + "\n")
        
        if not self.context.execution_log:
            print("⚠️ 暂无执行记录\n")
            return
        
        for log in self.context.execution_log:
            print(f"[步骤 {log['step_number']}] {log['function_name']}")
            print(f"   时间: {log['timestamp']}")
            print(f"   耗时: {log['execution_time']:.2f}秒")
            print(f"   参数: {log['arguments']}")
            
            if detailed:
                print(f"   结果预览:")
                preview = log['result_preview']
                for line in preview.split('\n')[:10]:  # 只显示前10行
                    print(f"      {line}")
                if log['result_length'] > 500:
                    print(f"      ... (共{log['result_length']}字符)")
            
            print()
        
        print(f"总计: {len(self.context.execution_log)} 个函数调用\n")
    
    def save_final_report(self, output_file: str = None):
        """
        保存最终报告到Markdown文件
        
        Args:
            output_file: 输出文件路径
        """
        if output_file is None:
            output_dir = "agent_reports"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"report_{self.context.session_id}.md")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Alpha Research Report\n\n")
            f.write(f"**Session ID**: {self.context.session_id}\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## 执行摘要\n\n")
            f.write(f"- 当前资产: {self.context.current_asset or 'N/A'}\n")
            f.write(f"- 数据状态: {'✅ 已加载' if self.context.market_data is not None else '❌ 未加载'}\n")
            f.write(f"- 分析项目: {len(self.context.analysis_results)}\n")
            f.write(f"- 可视化图表: {len(self.context.visualizations)}\n")
            f.write(f"- 因子建议: {len(self.context.factor_proposals)}\n")
            f.write(f"- 执行步骤: {len(self.context.execution_log)}\n\n")
            
            f.write(f"## 执行日志\n\n")
            for i, log in enumerate(self.context.execution_log, 1):
                f.write(f"### 步骤 {i}: {log['function_name']}\n\n")
                f.write(f"- **时间**: {log['timestamp']}\n")
                f.write(f"- **耗时**: {log['execution_time']:.2f}秒\n")
                f.write(f"- **参数**: `{log['arguments']}`\n")
                f.write(f"- **结果长度**: {log['result_length']} 字符\n\n")
                f.write(f"**结果预览**:\n```\n{log['result_preview']}\n```\n\n")
            
            if self.context.factor_proposals:
                f.write(f"## Alpha因子建议\n\n")
                for i, factor in enumerate(self.context.factor_proposals, 1):
                    f.write(f"### {i}. {factor['factor_name']}\n\n")
                    f.write(f"- **描述**: {factor['description']}\n")
                    f.write(f"- **公式**: `{factor['formula']}`\n")
                    f.write(f"- **理由**: {factor['justification']}\n")
                    f.write(f"- **预期表现**: Sharpe={factor['expected_performance']['expected_sharpe']:.2f}\n\n")
        
        print(f"📄 最终报告已保存至: {output_file}")
        return output_file
    
    def test_all_tools(self):
        """测试所有工具函数"""
        print("\n" + "="*70)
        print("🧪 测试Alpha Research Agent所有工具")
        print("="*70 + "\n")
        
        # 寻找测试文件
        test_dir = "qlib_data/etf_backup"
        csv_file = None
        
        if os.path.exists(test_dir):
            for fname in os.listdir(test_dir):
                if fname.endswith(".csv"):
                    csv_file = os.path.join(test_dir, fname)
                    break
        
        if not csv_file:
            print("❌ 未找到测试CSV文件")
            return
        
        print(f"📁 使用测试文件: {csv_file}\n")
        
        # 运行完整工作流
        result = self.run_complete_workflow(
            csv_file,
            user_input="请进行全面的技术分析和因子建议"
        )
        
        print("\n" + "="*70)
        print("📊 工作流执行结果:")
        print("="*70)
        print(result)
        print("\n" + "="*70)
        print("✅ 测试完成!")
        print("="*70 + "\n")
        
        # 打印执行日志
        self.print_execution_log(detailed=False)
        
        # 保存最终报告
        report_file = self.save_final_report()
        print(f"\n📄 详细报告: {report_file}\n")


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """主程序"""
    # 创建agent实例
    agent = AlphaResearchAgent()
    
    # 运行测试
    agent.test_all_tools()


if __name__ == "__main__":
    main()
