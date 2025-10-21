"""
Alpha Research Agent - 基于 OpenAI Agents SDK 构建 (修正版)
支持新版 openai>=1.0 API 和自动 context 传递
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from pydantic import BaseModel
import numpy as np
import qlib
from qlib.data import D
from qlib.contrib.data.handler import Alpha158

# ✅ 引入修复后的 Agent
from agents import Agent, function_tool

# 你的自定义工具模块（请确保路径正确）
from alpha_analysis_toolkit import AlphaAnalysisToolkit
from alpha_visualization_toolkit import AlphaVisualizationToolkit


# ============================================================================
# 数据结构定义
# ============================================================================

class FactorPerformance(BaseModel):
    """因子预期表现"""
    market_regime: str
    confidence_level: float
    expected_sharpe: float

    class Config:
        extra = "forbid"


class FactorProposal(BaseModel):
    """单个因子建议"""
    factor_name: str
    description: str
    formula: str
    justification: str
    expected_performance: FactorPerformance

    class Config:
        extra = "forbid"


class AlphaFactorResponse(BaseModel):
    """LLM结构化因子建议响应"""
    factor_proposals: List[FactorProposal]
    market_summary: str
    risk_assessment: str

    class Config:
        extra = "forbid"


# ============================================================================
# Context类：存储执行状态
# ============================================================================

class AlphaResearchContext:
    """Alpha研究上下文"""
    def __init__(self):
        self.current_asset: Optional[str] = None
        self.market_data: Optional[pd.DataFrame] = None
        self.analysis_results: Dict[str, Any] = {}
        self.visualizations: Dict[str, Any] = {}
        self.factor_proposals: List[Dict] = []
        self.iteration_count: int = 0
        self.execution_log: List[Dict[str, Any]] = []
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_function_call(self, name: str, args: Dict, result: str, execution_time: float = 0):
        self.execution_log.append({
            'timestamp': datetime.now().isoformat(),
            'function_name': name,
            'arguments': args,
            'result_preview': result[:500],
            'result_length': len(result),
            'execution_time': execution_time,
            'step_number': len(self.execution_log) + 1
        })

    def save_session_log(self, output_dir="agent_logs"):
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"session_{self.session_id}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": self.session_id,
                "execution_log": self.execution_log,
                "summary": {
                    "steps": len(self.execution_log),
                    "asset": self.current_asset,
                    "data_loaded": self.market_data is not None,
                    "factor_count": len(self.factor_proposals)
                }
            }, f, ensure_ascii=False, indent=2)
        return log_path


# ============================================================================
# 工具函数（装饰为 function_tool）
# ============================================================================

@function_tool
def load_and_analyze_data(ctx: AlphaResearchContext, csv_path: str, qlib_format: bool = False):
    """加载并分析资产数据，计算技术指标和信号"""
    start = datetime.now()
    try:
        print(f"🔧 加载数据: {csv_path}")
        data = AlphaAnalysisToolkit.load_asset_data(csv_path, data_format="csv")
        data = AlphaAnalysisToolkit.preprocess_data(data, qlib_format=qlib_format)
        indicators = AlphaAnalysisToolkit.calculate_technical_indicators(data)
        signals = AlphaAnalysisToolkit.generate_alpha_signals(data)
        risk = AlphaAnalysisToolkit.calculate_risk_metrics(data)

        ctx.current_asset = csv_path
        ctx.market_data = data
        ctx.analysis_results = {
            "technical_indicators": indicators,
            "signals": signals,
            "risk_metrics": risk
        }

        summary = (
            f"✅ 数据加载成功: {csv_path}\n"
            f"📊 行数={len(data)} | 指标={len(indicators)} | 信号={len(signals)}\n"
            f"⚠️ 年化波动率={risk.get('volatility', 0):.2%} | Sharpe={risk.get('sharpe_ratio', 0):.2f}"
        )

        ctx.log_function_call("load_and_analyze_data", {"csv_path": csv_path}, summary,
                              (datetime.now() - start).total_seconds())
        return summary
    except Exception as e:
        return f"❌ 加载失败: {e}"


@function_tool
def load_qlib_factors(ctx: AlphaResearchContext,
                      top_n: int = 30,
                      start_date: str = "2022-08-16",
                      end_date: str = "2024-12-31"):
    """
    用 Alpha158 Handler 计算并评估 IC/IR（正确方式）
    """
    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler import Alpha158

    start_time = datetime.now()
    try:
        provider_uri = "/content/AgenticTradng/qlib_data/stock_custom_day"
        feats_dir = os.path.join(provider_uri, "features")
        inst_list = sorted([d for d in os.listdir(feats_dir)
                            if os.path.isdir(os.path.join(feats_dir, d))])
        print(f"检测到 {len(inst_list)} 支股票: {inst_list[:5]} ...")

        # === 1) 构建 DatasetH ===
        handler_cfg = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": start_date,
                "end_time": end_date,
                "fit_start_time": start_date,
                "fit_end_time": end_date,
                "instruments": inst_list,
            },
        }
        ds = DatasetH(handler=handler_cfg, segments={"all": (start_date, end_date)})

        feat_df = ds.prepare("all", col_set="feature")
        valid_cols = [c for c in feat_df.columns
                      if feat_df[c].dropna().std() > 0 and feat_df[c].notna().sum() > 50]
        feat_df = feat_df[valid_cols]
        print(f"有效因子数: {len(valid_cols)}")

        # === 2) 未来一天收益 ===
        close = (D.features(instruments=inst_list, fields=["$close"],
                            start_time=start_date, end_time=end_date, freq="day")
                 .reset_index().set_index(["datetime", "instrument"]))
        ret_fwd = close["$close"].groupby(level=1).pct_change().shift(-1).rename("ret_fwd")

        panel = feat_df.join(ret_fwd, how="inner").replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_fwd"])
        gb = panel.groupby(level=0)

        # === 3) 计算 IC/IR ===
        records = []
        for f in valid_cols:
            ic_by_day = gb.apply(lambda g: g[f].corr(g["ret_fwd"], method="spearman")).dropna()
            if len(ic_by_day) < 10:
                continue
            ic_mean, ic_std = ic_by_day.mean(), ic_by_day.std(ddof=1)
            ir = ic_mean / ic_std if ic_std > 0 else np.nan
            records.append({"factor": f, "IC": float(ic_mean), "IR": float(ir), "days": len(ic_by_day)})

        perf_df = pd.DataFrame(records).dropna().sort_values("IR", ascending=False)
        ctx.analysis_results["qlib_factors"] = perf_df

        summary = f"""
✅ 成功计算 {len(perf_df)} 个 Alpha158 因子表现
前5名（按IR排序）:
{perf_df.head(5).to_string(index=False)}
"""
        ctx.log_function_call("load_qlib_factors",
                              {"provider": provider_uri, "top_n": top_n},
                              summary, (datetime.now() - start_time).total_seconds())
        print(summary)
        return summary

    except Exception as e:
        error_msg = f"❌ 加载 Alpha158 因子失败: {e}"
        ctx.log_function_call("load_qlib_factors", {}, error_msg, 0)
        print(error_msg)
        return error_msg



@function_tool
def propose_alpha_factors(ctx: AlphaResearchContext):
    """根据Qlib结果提出Alpha因子建议"""
    perf_df = ctx.analysis_results.get("qlib_factors")
    if perf_df is None or perf_df.empty:
        return "⚠️ 请先运行 load_qlib_factors()"
    ctx.factor_proposals = perf_df.head(5).to_dict("records")
    result = "\n".join(
        [f"{i+1}. {r['factor']} | IC={r['IC']:.3f} | IR={r['IR']:.2f}"
         for i, r in enumerate(ctx.factor_proposals)]
    )
    ctx.log_function_call("propose_alpha_factors", {}, result, 0)
    return result


@function_tool
def generate_iteration_report(ctx: AlphaResearchContext, iteration_number: int = 1):
    """生成迭代报告"""
    ctx.iteration_count = iteration_number
    report = f"""
📋 研究报告 #{iteration_number}
资产: {ctx.current_asset}
分析结果: {len(ctx.analysis_results)}
可视化图表: {len(ctx.visualizations)}
因子建议: {len(ctx.factor_proposals)}
执行步骤: {len(ctx.execution_log)}
"""
    path = ctx.save_session_log()
    report += f"📄 日志保存: {path}"
    ctx.log_function_call("generate_iteration_report", {"iteration_number": iteration_number}, report, 0)
    return report


# ============================================================================
# Alpha Research Agent 主类
# ============================================================================

class AlphaResearchAgent:
    """Alpha研究智能体 - 执行完整Alpha分析流程"""

    def __init__(self):
        self.context = AlphaResearchContext()
        self.agent = Agent(
            name="AlphaResearchAgent",
            model="gpt-4o-mini",
            instructions="""
你是一名专业的量化研究助理。你可以调用工具:
- load_and_analyze_data
- load_qlib_factors
- propose_alpha_factors
- generate_iteration_report
请基于数据进行完整分析。
""",
            tools=[
                load_and_analyze_data,
                load_qlib_factors,
                propose_alpha_factors,
                generate_iteration_report,
            ],
        )

    async def run_analysis(self, user_request: str):
        """异步运行分析任务"""
        print(f"\n🚀 开始Alpha研究分析 | Session {self.context.session_id}\n")
        result_text = self.agent.run(user_request, context=self.context)
        log_path = self.context.save_session_log()
        print(f"\n✅ 分析完成 | 日志: {log_path}\n")
        return result_text

    def run_analysis_sync(self, user_request: str):
        """同步执行版本"""
        return asyncio.run(self.run_analysis(user_request))

    def summarize_with_llm(self):
        """
        使用 LLM 对当前 context 的分析结果进行总结。
        """
        # 从 context 里提取关键信息
        risk = self.context.analysis_results.get("risk_metrics", {})
        factor_df = self.context.analysis_results.get("qlib_factors")

        # 生成提示词（prompt）
        text_prompt = f"""
你是一名量化研究助理。以下是一次Alpha研究任务的结果。

【1️⃣ 数据分析结果】
年化波动率: {risk.get('volatility', '未知')}
夏普比率: {risk.get('sharpe_ratio', '未知')}
其它指标: {risk}

【2️⃣ Alpha因子表现（前10个）】
{factor_df.head(10).to_string(index=False) if factor_df is not None else '暂无因子数据'}

请你帮我完成以下分析：
- 简要总结市场特征；
- 分析这些因子的代表意义；
- 指出哪些因子组合可能有用；
- 给出改进方向或进一步研究建议。
"""

        print("\n🧠 调用 LLM 进行分析总结 ...")

        # 使用 OpenAI 客户端调用模型
        try:
            response = self.agent.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一位专业的量化研究分析师，擅长Alpha因子分析。"},
                    {"role": "user", "content": text_prompt},
                ],
            )
            summary_text = response.choices[0].message.content
            print("\n📊 LLM 分析结果:\n", summary_text[:1500])
            return summary_text
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return f"❌ LLM 调用失败: {e}"

    def run_complete_workflow(self, csv_path: str, user_input: str = ""):
        """
        完整执行Alpha研究流程：
        1. 调用工具执行分析；
        2. 调用LLM进行总结分析；
        3. 输出完整报告。
        """
        prompt = f"""
请对 {csv_path} 进行完整alpha研究，包括:
1. 调用 load_and_analyze_data
2. 调用 load_qlib_factors
3. 调用 propose_alpha_factors
4. 生成第1次迭代报告
{user_input}
"""

        # === (1) 执行分析 ===
        base_result = self.run_analysis_sync(prompt)

        # === (2) 让 LLM 总结 ===
        llm_summary = self.summarize_with_llm()

        # === (3) 合并输出 ===
        final_report = base_result + "\n\n======\n📈 LLM总结分析：\n" + llm_summary
        print("\n✅ 完整报告生成完成。")
        return final_report


# ============================================================================
# 主入口
# ============================================================================

def main():
    """测试入口"""
    import qlib
    qlib.init(provider_uri="/content/AgenticTradng/qlib_data/stock_custom_day", region="us")
    agent = AlphaResearchAgent()
    report = agent.run_complete_workflow(
        "/content/AgenticTradng/qlib_data/stock_backup/XOM_daily.csv",
        user_input="进行技术分析与因子建议"
    )

    # 打印最终总结
    print("\n\n============================")
    print("📄 最终综合报告（含LLM分析）")
    print("============================\n")
    print(report)



if __name__ == "__main__":
    main()
