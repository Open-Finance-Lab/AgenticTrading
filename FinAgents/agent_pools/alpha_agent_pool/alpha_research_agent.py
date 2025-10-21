"""
Alpha Research Agent - Built on OpenAI Agents SDK (Revised Version)
Supports openai>=1.0 API and automatic context passing.
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

# ✅ Import the revised Agent class
from agents import Agent, function_tool

# Custom toolkits (ensure correct import paths)
from alpha_analysis_toolkit import AlphaAnalysisToolkit
from alpha_visualization_toolkit import AlphaVisualizationToolkit


# ============================================================================
# Data Structure Definitions
# ============================================================================

class FactorPerformance(BaseModel):
    """Expected factor performance"""
    market_regime: str
    confidence_level: float
    expected_sharpe: float

    class Config:
        extra = "forbid"


class FactorProposal(BaseModel):
    """Single factor proposal"""
    factor_name: str
    description: str
    formula: str
    justification: str
    expected_performance: FactorPerformance

    class Config:
        extra = "forbid"


class AlphaFactorResponse(BaseModel):
    """Structured response for factor proposals"""
    factor_proposals: List[FactorProposal]
    market_summary: str
    risk_assessment: str

    class Config:
        extra = "forbid"


# ============================================================================
# Context Class
# ============================================================================

class AlphaResearchContext:
    """Alpha research execution context"""
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
# Tool Functions (decorated with @function_tool)
# ============================================================================

@function_tool
def load_and_analyze_data(ctx: AlphaResearchContext, csv_path: str, qlib_format: bool = False):
    """Load and analyze asset data, compute technical indicators and signals"""
    start = datetime.now()
    try:
        print(f"Loading data: {csv_path}")
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
            f"Data loaded successfully: {csv_path}\n"
            f"Rows={len(data)} | Indicators={len(indicators)} | Signals={len(signals)}\n"
            f"Volatility={risk.get('volatility', 0):.2%} | Sharpe={risk.get('sharpe_ratio', 0):.2f}"
        )

        ctx.log_function_call("load_and_analyze_data", {"csv_path": csv_path}, summary,
                              (datetime.now() - start).total_seconds())
        return summary
    except Exception as e:
        return f"Data loading failed: {e}"


@function_tool
def load_qlib_factors(ctx: AlphaResearchContext,
                      top_n: int = 30,
                      start_date: str = "2022-08-16",
                      end_date: str = "2024-12-31"):
    """
    Compute and evaluate IC/IR using Alpha158 handler
    """
    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler import Alpha158

    start_time = datetime.now()
    try:
        provider_uri = "/content/AgenticTradng/qlib_data/stock_custom_day"
        feats_dir = os.path.join(provider_uri, "features")
        inst_list = sorted([d for d in os.listdir(feats_dir)
                            if os.path.isdir(os.path.join(feats_dir, d))])
        print(f"Detected {len(inst_list)} stocks: {inst_list[:5]} ...")

        # === 1) Build DatasetH ===
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
        print(f"Valid factors: {len(valid_cols)}")

        # === 2) Next-day return ===
        close = (D.features(instruments=inst_list, fields=["$close"],
                            start_time=start_date, end_time=end_date, freq="day")
                 .reset_index().set_index(["datetime", "instrument"]))
        ret_fwd = close["$close"].groupby(level=1).pct_change().shift(-1).rename("ret_fwd")

        panel = feat_df.join(ret_fwd, how="inner").replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_fwd"])
        gb = panel.groupby(level=0)

        # === 3) Compute IC/IR ===
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
Successfully computed {len(perf_df)} Alpha158 factor performances
Top 5 (by IR):
{perf_df.head(5).to_string(index=False)}
"""
        ctx.log_function_call("load_qlib_factors",
                              {"provider": provider_uri, "top_n": top_n},
                              summary, (datetime.now() - start_time).total_seconds())
        print(summary)
        return summary

    except Exception as e:
        error_msg = f"Failed to load Alpha158 factors: {e}"
        ctx.log_function_call("load_qlib_factors", {}, error_msg, 0)
        print(error_msg)
        return error_msg


@function_tool
def propose_alpha_factors(ctx: AlphaResearchContext):
    """Propose alpha factors based on Qlib results"""
    perf_df = ctx.analysis_results.get("qlib_factors")
    if perf_df is None or perf_df.empty:
        return "Please run load_qlib_factors() first."
    ctx.factor_proposals = perf_df.head(5).to_dict("records")
    result = "\n".join(
        [f"{i+1}. {r['factor']} | IC={r['IC']:.3f} | IR={r['IR']:.2f}"
         for i, r in enumerate(ctx.factor_proposals)]
    )
    ctx.log_function_call("propose_alpha_factors", {}, result, 0)
    return result


@function_tool
def generate_iteration_report(ctx: AlphaResearchContext, iteration_number: int = 1):
    """Generate iteration report"""
    ctx.iteration_count = iteration_number
    report = f"""
Research Report #{iteration_number}
Asset: {ctx.current_asset}
Analysis results: {len(ctx.analysis_results)}
Visualizations: {len(ctx.visualizations)}
Factor proposals: {len(ctx.factor_proposals)}
Execution steps: {len(ctx.execution_log)}
"""
    path = ctx.save_session_log()
    report += f"Log saved: {path}"
    ctx.log_function_call("generate_iteration_report", {"iteration_number": iteration_number}, report, 0)
    return report


# ============================================================================
# Alpha Research Agent
# ============================================================================

class AlphaResearchAgent:
    """Alpha Research Agent - executes a full alpha analysis workflow"""

    def __init__(self):
        self.context = AlphaResearchContext()
        self.agent = Agent(
            name="AlphaResearchAgent",
            model="gpt-4o-mini",
            instructions="""
You are a professional quantitative research assistant. You can call the following tools:
- load_and_analyze_data
- load_qlib_factors
- propose_alpha_factors
- generate_iteration_report
Please perform a full alpha research workflow.
""",
            tools=[
                load_and_analyze_data,
                load_qlib_factors,
                propose_alpha_factors,
                generate_iteration_report,
            ],
        )

    async def run_analysis(self, user_request: str):
        """Run the analysis asynchronously"""
        print(f"\nStarting Alpha Research Analysis | Session {self.context.session_id}\n")
        result_text = self.agent.run(user_request, context=self.context)
        log_path = self.context.save_session_log()
        print(f"\nAnalysis complete | Log: {log_path}\n")
        return result_text

    def run_analysis_sync(self, user_request: str):
        """Synchronous execution"""
        return asyncio.run(self.run_analysis(user_request))

    def summarize_with_llm(self):
        """
        Use the LLM to summarize current analysis results.
        """
        risk = self.context.analysis_results.get("risk_metrics", {})
        factor_df = self.context.analysis_results.get("qlib_factors")

        # Construct the LLM prompt
        text_prompt = f"""
You are a quantitative research assistant. Below are the results from an alpha research session.

[1] Data Analysis Results
Volatility: {risk.get('volatility', 'Unknown')}
Sharpe Ratio: {risk.get('sharpe_ratio', 'Unknown')}
Other metrics: {risk}

[2] Alpha Factor Performance (Top 10)
{factor_df.head(10).to_string(index=False) if factor_df is not None else 'No factor data available'}

Please provide:
- A short summary of market characteristics
- Analysis of what these factors represent
- Which combinations may be useful
- Suggestions for improvement or further research
"""

        print("\nCalling LLM for analysis summary ...")

        try:
            response = self.agent.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional quantitative analyst specializing in alpha factor research."},
                    {"role": "user", "content": text_prompt},
                ],
            )
            summary_text = response.choices[0].message.content
            print("\nLLM Summary:\n", summary_text[:1500])
            return summary_text
        except Exception as e:
            print(f"LLM call failed: {e}")
            return f"LLM call failed: {e}"

    def run_complete_workflow(self, csv_path: str, user_input: str = ""):
        """
        Execute a full alpha research workflow:
        1. Run analysis tools
        2. Use LLM for summarization
        3. Generate a final combined report
        """
        prompt = f"""
Perform a complete alpha research on {csv_path}, including:
1. Call load_and_analyze_data
2. Call load_qlib_factors
3. Call propose_alpha_factors
4. Generate the first iteration report
{user_input}
"""

        # Step 1: Run analysis
        base_result = self.run_analysis_sync(prompt)

        # Step 2: Summarize with LLM
        llm_summary = self.summarize_with_llm()

        # Step 3: Combine results
        final_report = base_result + "\n\n======\nLLM Summary:\n" + llm_summary
        print("\nFinal report generated successfully.")
        return final_report


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Test entry point"""
    import qlib
    qlib.init(provider_uri="/content/AgenticTradng/qlib_data/stock_custom_day", region="us")
    agent = AlphaResearchAgent()
    report = agent.run_complete_workflow(
        "/content/AgenticTradng/qlib_data/stock_backup/XOM_daily.csv",
        user_input="Perform technical analysis and factor proposal"
    )

    # Print final report
    print("\n\n============================")
    print("Final Combined Report (with LLM Summary)")
    print("============================\n")
    print(report)


if __name__ == "__main__":
    main()
