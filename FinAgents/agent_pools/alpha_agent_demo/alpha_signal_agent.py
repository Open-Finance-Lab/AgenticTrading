"""
Alpha Signal Agent using OpenAI Agent SDK

This agent integrates Qlib factor construction, technical indicators, and ML inference
to generate alpha trading signals. It supports both a monolithic pipeline and a 
flexible ReAct workflow where the agent can choose tools dynamically.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
alpha_agent_pool_path = parent_dir / "alpha_agent_pool"
sys.path.append(str(alpha_agent_pool_path))

# Import OpenAI Agents SDK
import nest_asyncio
nest_asyncio.apply()
from agents import Agent, Runner, function_tool, RunContextWrapper

# Import Qlib utilities
try:
    qlib_path = alpha_agent_pool_path / "qlib_local"
    if not qlib_path.exists():
        qlib_path = alpha_agent_pool_path / "qlib"
    sys.path.insert(0, str(qlib_path))
    from utils import QlibConfig, DataProcessor
    from data_interfaces import FactorInput
    from standard_factor_calculator import StandardFactorCalculator
except ImportError as e:
    print(f"Warning: Qlib modules not found: {e}. Some features may be limited.")
    from dataclasses import dataclass, field
    from typing import List
    
    @dataclass
    class QlibConfig:
        provider_uri: str = ""
        instruments: List[str] = field(default_factory=list)
    
    class DataProcessor:
        def __init__(self, config):
            self.config = config
        def add_returns(self, data):
            return data
        def create_technical_features(self, data):
            return data
    
    @dataclass
    class FactorInput:
        factor_name: str = ""
        factor_type: str = ""
        calculation_method: str = ""
        expression: str = None
        function_name: str = None
        lookback_period: int = 20
    
    class StandardFactorCalculator:
        def calculate(self, data, factor_input):
            return pd.Series(dtype=float)

# Import ML libraries
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Warning: ML libraries not found. Install sklearn and lightgbm for full functionality.")


# ==============================
# Internal Implementation (Pure Logic)
# ==============================

def _calculate_technical_indicators(data: pd.DataFrame, indicators: List[str]) -> Dict[str, Any]:
    """Internal technical indicator calculation."""
    try:
        # Identify price columns
        close_col = None
        for col in ['$close', 'close', 'Close']:
            if col in data.columns:
                close_col = col
                break
        if close_col is None:
            return {"status": "error", "message": "No close price column found"}
        
        results = {}
        
        if 'RSI' in indicators:
            delta = data[close_col].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            results['RSI'] = (100 - (100 / (1 + rs))).to_dict()
            
        if 'MACD' in indicators:
            ema_fast = data[close_col].ewm(span=12).mean()
            ema_slow = data[close_col].ewm(span=26).mean()
            macd = ema_fast - ema_slow
            results['MACD'] = macd.to_dict()
            
        if 'Bollinger' in indicators:
            ma = data[close_col].rolling(window=20).mean()
            std = data[close_col].rolling(window=20).std()
            results['Bollinger_upper'] = (ma + 2*std).to_dict()
            results['Bollinger_lower'] = (ma - 2*std).to_dict()
            
        return {"status": "success", "indicators": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _prepare_features_targets(data: pd.DataFrame, indicators: List[str]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Helper to calculate features and targets from raw data."""
    # 1. Preprocessing
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()
    data = data.reset_index(drop=True)
    
    # Normalize columns
    col_map = {c: c.lower() for c in data.columns}
    data = data.rename(columns=col_map)
    if 'instrument' in data.columns: data = data.rename(columns={'instrument': 'symbol'})
    if 'datetime' in data.columns: data = data.rename(columns={'datetime': 'date'})
    
    if 'close' not in data.columns:
        raise ValueError("Missing close column")

    # 2. Indicators
    ind_res = _calculate_technical_indicators(data, indicators)
    features = pd.DataFrame(index=data.index)
    
    if ind_res['status'] == 'success':
        for name, vals in ind_res['indicators'].items():
            features[name] = pd.Series(vals)
    else:
        raise ValueError(f"Indicator calc failed: {ind_res.get('message')}")
        
    # 3. Target (Forward Return)
    if 'symbol' in data.columns:
        targets = data.groupby('symbol')['close'].pct_change().shift(-1)
    else:
        targets = data['close'].pct_change().shift(-1)
        
    return features, targets

def _train_model_and_predict(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    model_type: str
) -> Dict[str, Any]:
    """
    Train on X_train/y_train, Predict on X_test.
    NO LEAKAGE: Model sees only train data.
    """
    try:
        # Align Train Data
        train_aligned = pd.concat([X_train, y_train], axis=1).dropna()
        if len(train_aligned) < 10: # Allow small samples for demo
             # If train is empty, maybe fallback to random or error?
             pass
        
        if train_aligned.empty:
             return {"status": "error", "message": "Insufficient training data (after alignment/drop na)"}

        X_train_clean = train_aligned.iloc[:, :-1]
        y_train_clean = train_aligned.iloc[:, -1]
        
        # Handle Test Data (fill NaNs in features with 0 or drop?)
        # For prediction, we shouldn't drop rows if possible, but if features are NaN (e.g. first 14 days), we can't predict.
        # So we fillNa or drop.
        X_test_clean = X_test.fillna(0) # Simple imputation
        
        preds_series = pd.Series(index=X_test.index, dtype=float)
        
        model = None
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        else:
            return {"status": "error", "message": f"Unknown model {model_type}"}
            
        # FIT on TRAIN
        model.fit(X_train_clean, y_train_clean)
        
        # PREDICT on TEST
        preds = model.predict(X_test_clean)
        preds_series.loc[X_test_clean.index] = preds
        
        return {
            "status": "success",
            "predictions": preds_series.to_dict(),
            "model_info": {"type": model_type}
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _run_alpha_pipeline_impl(
    test_data: pd.DataFrame,
    train_data: Optional[pd.DataFrame],
    factors: List[Dict],
    indicators: List[str],
    model_type: str,
    signal_threshold: float,
    data_processor: Any
) -> Dict[str, Any]:
    """Full pipeline implementation (Macro)."""
    try:
        # If no train data provided, fallback to In-Sample (Old behavior) or Error?
        # For backwards compatibility, if train_data is None, split test_data or use it as both (with warning)
        if train_data is None or train_data.empty:
            print("WARNING: No training data provided. Using In-Sample training (Leakage Risk).")
            train_data = test_data
            
        # 1. Prepare Features & Targets
        try:
            X_train, y_train = _prepare_features_targets(train_data, indicators)
            X_test, _ = _prepare_features_targets(test_data, indicators)
        except Exception as e:
            return {"status": "error", "message": f"Feature Prep Failed: {str(e)}"}
            
        # 2. Train & Predict
        model_res = _train_model_and_predict(X_train, y_train, X_test, model_type)
        if model_res['status'] != 'success':
            return model_res
            
        # 3. Signals
        preds = pd.Series(model_res['predictions'])
        signals = preds.apply(lambda x: 1.0 if x > signal_threshold else (-1.0 if x < -signal_threshold else 0.0))
        
        # Restore index to (date, symbol) for output
        # Need to normalize test_data first to match 'date', 'symbol' expectations
        # We can reuse the logic from _prepare_features_targets roughly
        data_norm = test_data.copy()
        if isinstance(data_norm.index, pd.MultiIndex): data_norm = data_norm.reset_index()
        col_map = {c: c.lower() for c in data_norm.columns}
        data_norm = data_norm.rename(columns=col_map)
        if 'instrument' in data_norm.columns: data_norm = data_norm.rename(columns={'instrument': 'symbol'})
        if 'datetime' in data_norm.columns: data_norm = data_norm.rename(columns={'datetime': 'date'})
        
        # Ensure index matches preds
        # preds index matches X_test index, which matches test_data (reset_index(drop=True))
        # So we need to map back using integer position?
        # Yes, X_test index is RangeIndex from reset_index(drop=True) inside helper.
        # We applied same reset to data_norm (but need drop=True)
        data_norm = data_norm.reset_index(drop=True)
        
        # Align
        signals.index = data_norm.index
        
        if 'date' in data_norm.columns and 'symbol' in data_norm.columns:
            signals.index = pd.MultiIndex.from_frame(data_norm[['date', 'symbol']])
        elif 'date' in data_norm.columns:
            signals.index = pd.Index(data_norm['date'])
            
        return {
            "status": "success",
            "signals": signals.to_dict(),
            "model_performance": model_res.get('model_info')
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ==============================
# Tool Definitions (Context-Aware)
# ==============================

@function_tool
def run_alpha_pipeline(ctx: RunContextWrapper[Any]) -> str:
    """
    Execute the complete standard alpha pipeline (Calculate Indicators -> Train Model -> Generate Signals).
    Use this for a quick, standard analysis.
    """
    print("DEBUG: 🛠️ run_alpha_pipeline (Macro) INVOKED")
    try:
        data = ctx.context.get('data')      # This is TEST data (current year)
        train_data = ctx.context.get('train_data') # This is TRAIN data (prev year)
        
        factors = ctx.context.get('factors', [])
        indicators = ctx.context.get('indicators', ['RSI', 'MACD'])
        model_type = ctx.context.get('model_type', 'linear')
        threshold = ctx.context.get('signal_threshold', 0.0)
        data_processor = ctx.context.get('data_processor')
        
        if data is None: return "Error: No test data in context."
        
        result = _run_alpha_pipeline_impl(data, train_data, factors, indicators, model_type, threshold, data_processor)
        ctx.context['result'] = result
        return f"Pipeline completed. Status: {result.get('status')}"
    except Exception as e:
        return f"Error: {e}"

@function_tool
def calculate_indicators_tool(ctx: RunContextWrapper[Any], indicators: List[str]) -> str:
    """
    Calculate specific technical indicators on the current data (Test Data).
    """
    print(f"DEBUG: 🛠️ calculate_indicators_tool INVOKED with {indicators}")
    try:
        data = ctx.context.get('data')
        if data is None: return "Error: No data in context."
        
        res = _calculate_technical_indicators(data, indicators)
        if res['status'] == 'success':
            # Store features in context
            if 'features' not in ctx.context:
                ctx.context['features'] = pd.DataFrame(index=data.index)
            
            features = ctx.context['features']
            for name, vals in res['indicators'].items():
                features[name] = pd.Series(vals)
            ctx.context['features'] = features
            return f"Calculated {len(res['indicators'])} indicators for Test Set."
        return f"Failed: {res.get('message')}"
    except Exception as e:
        return f"Error: {e}"

@function_tool
def train_predict_tool(ctx: RunContextWrapper[Any], model_type: str = "linear") -> str:
    """
    Train a model using TRAINING data and predict on CURRENT features.
    Requires 'train_data' in context.
    """
    print(f"DEBUG: 🛠️ train_predict_tool INVOKED with {model_type}")
    try:
        test_data = ctx.context.get('data')
        test_features = ctx.context.get('features')
        train_data = ctx.context.get('train_data')
        indicators = ctx.context.get('indicators', ['RSI', 'MACD']) # Need to know which indicators used
        
        if test_features is None or test_features.empty:
            return "Error: Calculate indicators for test data first."
            
        if train_data is None or train_data.empty:
            return "Error: No training data provided for rolling window."
            
        # Prepare Train Features (Calculate same indicators on train data)
        try:
            X_train, y_train = _prepare_features_targets(train_data, indicators)
        except Exception as e:
            return f"Error preparing training data: {e}"
            
        # Train & Predict
        res = _train_model_and_predict(X_train, y_train, test_features, model_type)
        
        if res['status'] == 'success':
            ctx.context['raw_predictions'] = res['predictions']
            return "Model trained on historical data and predictions generated for current period."
        return f"Training failed: {res.get('message')}"
    except Exception as e:
        return f"Error: {e}"

@function_tool
def submit_signals_tool(ctx: RunContextWrapper[Any], threshold: float = 0.0) -> str:
    """
    Convert predictions to trading signals and finalize the task.
    """
    print(f"DEBUG: 🛠️ submit_signals_tool INVOKED")
    try:
        preds_dict = ctx.context.get('raw_predictions')
        if not preds_dict: return "Error: No predictions found."
        
        data = ctx.context.get('data')
        preds = pd.Series(preds_dict)
        
        signals = preds.apply(lambda x: 1.0 if x > threshold else (-1.0 if x < -threshold else 0.0))
        
        # Format index
        if 'date' in data.columns and 'symbol' in data.columns:
            # Re-normalize if needed
            # Assuming data hasn't changed order since indicator calc
            # Ideally we use index alignment but data usually has RangeIndex or original
            # We trust the order matches if preds_dict came from same process
            pass
        
        # Robust Index Alignment
        # If preds index is integer (from reset_index), we map it back?
        # Or we just trust that 'data' is the source of truth
        
        # Simple approach:
        signals.index = data.index # Align with original data index if preserved
        
        # If data columns need normalizing to find date/symbol
        data_copy = data.copy()
        if isinstance(data_copy.index, pd.MultiIndex): data_copy = data_copy.reset_index()
        data_copy.columns = [str(c).lower() for c in data_copy.columns]
        if 'datetime' in data_copy.columns: data_copy = data_copy.rename(columns={'datetime': 'date'})
        if 'instrument' in data_copy.columns: data_copy = data_copy.rename(columns={'instrument': 'symbol'})
        
        if 'date' in data_copy.columns and 'symbol' in data_copy.columns:
             signals.index = pd.MultiIndex.from_frame(data_copy[['date', 'symbol']])
        elif 'date' in data_copy.columns:
             signals.index = pd.Index(data_copy['date'])
            
        ctx.context['result'] = {
            "status": "success",
            "signals": signals.to_dict()
        }
        return "Signals generated and submitted. Task complete."
    except Exception as e:
        return f"Error: {e}"


# ==============================
# Alpha Signal Agent
# ==============================

class AlphaSignalAgent:
    def __init__(
        self,
        name: str = "AlphaSignalAgent",
        model: str = "gpt-4o",
        qlib_config: Optional[QlibConfig] = None
    ):
        self.name = name
        self.model = model
        self.qlib_config = qlib_config or QlibConfig()
        self.data_processor = DataProcessor(self.qlib_config)
        
        # Register context-aware tools
        self.tools = [
            run_alpha_pipeline,          # Macro tool (easy mode)
            calculate_indicators_tool,   # Granular tool
            train_predict_tool,          # Granular tool
            submit_signals_tool          # Granular tool
        ]
        
        self.agent = Agent(
            name=name,
            instructions="""
            You are an Alpha Signal Agent.
            You have market data in your context:
            - 'data': Current period data (Test Set) to generate signals for.
            - 'train_data': Previous period data (Train Set) to train models on.
            
            You can choose ONE of two paths:
            1. FAST PATH: Use 'run_alpha_pipeline' to execute a standard strategy immediately.
            2. CUSTOM PATH: Build a strategy step-by-step:
               a. Call 'calculate_indicators_tool' (e.g. with ['RSI', 'MACD']) - calculates features on Test Data.
               b. Call 'train_predict_tool' (e.g. 'random_forest') - trains on Train Data, predicts on Test Data.
               c. Call 'submit_signals_tool'
            
            Choose the Custom Path if you need to refine the model or use specific indicators.
            """,
            model=model,
            tools=self.tools
        )
    
    def run(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> str:
        return self.agent.run(user_request, context=context, max_turns=10)
    
    def generate_signals_from_data(
        self,
        data: pd.DataFrame,
        factors: List[Dict[str, Any]],
        indicators: List[str],
        model_type: str = "linear",
        signal_threshold: float = 0.0,
        train_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate signals using LLM-based execution.
        """
        # Prepare context
        context = {
            'data': data,              # Test Data
            'train_data': train_data,  # Train Data
            'factors': factors,
            'indicators': indicators,
            'model_type': model_type,
            'signal_threshold': signal_threshold,
            'data_processor': self.data_processor
        }
        
        print("DEBUG: 🤖 Requesting Alpha Agent LLM...")
        
        # We instruct the agent. The prompt optimization can now change this instruction!
        # Default instruction uses the variables passed in.
        # But the agent's system prompt (self.agent.instructions) encourages flexibility.
        
        request = f"Generate alpha signals. Default suggestion: Use indicators {indicators} and model {model_type}."
        
        result = Runner.run_sync(self.agent, request, context=context)
        print(f"DEBUG: LLM finished. Context keys: {list(context.keys())}")
        
        if 'result' in context:
            return context['result']
        
        return {'status': 'error', 'message': 'LLM did not produce a result in context'}

if __name__ == "__main__":
    print("Alpha Signal Agent Initialized")
