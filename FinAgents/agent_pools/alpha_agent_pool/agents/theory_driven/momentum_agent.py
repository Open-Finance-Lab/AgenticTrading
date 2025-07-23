"""
Entry point for starting the agent, supports direct command-line execution.
All comments in this file are in English for clarity and maintainability.
"""

# agent_pools/alpha_agent_pool/agents/theory_driven/momentum_agent.py

from mcp.server.fastmcp import FastMCP, Context as MCPContext
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from schema.theory_driven_schema import (
    MomentumAgentConfig, MomentumSignalRequest, AlphaStrategyFlow, MarketContext, Decision, Action, PerformanceFeedback, Metadata
)
from .a2a_client import AlphaAgentA2AClient, create_alpha_pool_a2a_client, A2AProtocolError
from typing import List, Dict, Any, Optional
import asyncio
import sys
import argparse
import json
import os
from datetime import datetime
import hashlib
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path}")
except ImportError:
    print("⚠️ python-dotenv not installed. Please install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Failed to load .env file: {e}")

# LLM Integration
try:
    import openai
    from openai import AsyncOpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    AsyncOpenAI = None


# Configure logging: always overwrite log file on agent start
log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../momentum_agent.log'))
if os.path.exists(log_path):
    try:
        os.remove(log_path)
    except Exception:
        pass
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MomentumAgent:
    def run_rl_backtest_and_update(self, strategy_flow_path: str, market_data_path: str, rl_method: str = "q_learning", learning_rate: float = 0.1, baseline: float = 0.0, feedback_days: int = 10, initial_cash: float = 100000.0):
        """
        Run a backtest using given strategy flow and market data, then update RL policy automatically.
        Args:
            strategy_flow_path (str): Path to strategy flow JSON file.
            market_data_path (str): Path to market data CSV file.
            rl_method (str): "q_learning" or "policy_gradient".
            learning_rate (float): RL learning rate.
            baseline (float): Baseline for policy gradient.
            feedback_days (int): Number of feedbacks to use for RL update.
            initial_cash (float): Initial cash for backtest.
        """
        import json
        import os
        import logging
        import matplotlib.pyplot as plt
        from datetime import datetime
        import numpy as np
        logger = logging.getLogger(__name__)
        # 1. Run backtest using Backtrader
        try:
            import pandas as pd
            import backtrader as bt
            import backtrader.feeds as btfeeds
            # Load signals
            with open(strategy_flow_path, 'r', encoding='utf-8') as f:
                flow = json.load(f)
            signals = []
            for entry in flow:
                try:
                    signal_info = entry['alpha_signals']['signals']['AAPL']
                    signal = signal_info['decision']['signal']
                    weight = signal_info['action'].get('execution_weight', signal_info['decision'].get('confidence', 0.0))
                    confidence = signal_info['decision'].get('confidence', 0.0)
                    predicted_return = signal_info['decision'].get('predicted_return', None)
                    selected_timeframe = signal_info.get('selected_timeframe', None)
                    signals.append({'signal': signal, 'execution_weight': weight, 'confidence': confidence, 'predicted_return': predicted_return, 'selected_timeframe': selected_timeframe})
                except Exception as e:
                    continue
            # Load market data
            market_df = pd.read_csv(market_data_path)
            # Use 'timestamp' if present, else 'date'
            if 'timestamp' in market_df.columns:
                market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
                market_df.set_index('timestamp', inplace=True)
                logger.info(f"[RL] Market data loaded with 'timestamp' column, {len(market_df)} rows.")
            elif 'date' in market_df.columns:
                market_df['date'] = pd.to_datetime(market_df['date'])
                market_df.set_index('date', inplace=True)
                logger.info(f"[RL] Market data loaded with 'date' column, {len(market_df)} rows.")
            else:
                logger.error("[RL] Market data missing both 'timestamp' and 'date' columns!")
                return None
            # Fill missing OHLCV columns with close price or 1
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in market_df.columns:
                    if 'price' in market_df.columns:
                        market_df[col] = market_df['price']
                    else:
                        market_df[col] = 1
            market_df['volume'] = market_df['volume'].fillna(1)
            bt_df = market_df[['open', 'high', 'low', 'close', 'volume']].copy()
            data = btfeeds.PandasData(dataname=bt_df)
            # Backtrader strategy
            class StrategyFlowBacktest(bt.Strategy):
                params = (('signal_data', None),)
                def __init__(self):
                    self.signal_data = self.params.signal_data
                    self.order = None
                    self.bar_index = 0
                    self.log_records = []
                    self.prev_value = None
                    self.trade_count = 0
                    self.win_count = 0
                    self.last_trade_price = None
                    self.equity_curve = []
                def next(self):
                    if self.bar_index >= len(self.signal_data):
                        self.bar_index += 1
                        return
                    signal_info = self.signal_data[self.bar_index]
                    self.bar_index += 1
                    signal = signal_info.get("signal", "HOLD")
                    confidence = signal_info.get("confidence", 0.0)
                    predicted_return = signal_info.get("predicted_return", None)
                    price = self.datas[0].close[0]
                    position = self.position.size
                    trade_executed = False
                    # 优化信号执行逻辑：只在持仓为0时BUY，持仓>0时SELL
                    if signal == "BUY" and position == 0:
                        size = int(self.broker.getvalue() / price)
                        if size > 0:
                            self.order = self.buy(size=size)
                            trade_executed = True
                            self.trade_count += 1
                            self.last_trade_price = price
                    elif signal == "SELL" and position > 0:
                        self.order = self.sell(size=position)
                        trade_executed = True
                        self.trade_count += 1
                        if self.last_trade_price is not None and price > self.last_trade_price:
                            self.win_count += 1
                        self.last_trade_price = None
                    # Calculate daily return
                    if self.prev_value is None:
                        self.prev_value = self.broker.getvalue()
                    cur_value = self.broker.getvalue()
                    daily_return = (cur_value - self.prev_value) / self.prev_value if self.prev_value else 0.0
                    self.prev_value = cur_value
                    self.equity_curve.append(cur_value)
                    self.log_records.append({
                        'bar_index': self.bar_index,
                        'signal': signal,
                        'confidence': confidence,
                        'predicted_return': predicted_return,
                        'price': price,
                        'position': position,
                        'target_ratio': None,
                        'portfolio_value': cur_value,
                        'daily_return': daily_return,
                        'selected_timeframe': signal_info.get('selected_timeframe', None),
                        'trade_executed': trade_executed
                    })
                def stop(self):
                    predicted_returns = []
                    actual_returns = []
                    confidences = []
                    for rec in self.log_records:
                        if rec['predicted_return'] is not None:
                            predicted_returns.append(rec['predicted_return'])
                            actual_returns.append(rec['daily_return'])
                            confidences.append(rec['confidence'])
                    ic = np.corrcoef(predicted_returns, actual_returns)[0,1] if len(predicted_returns) > 1 else None
                    ic_mean = np.mean(ic) if ic is not None else None
                    ic_std = np.std(ic) if ic is not None else None
                    ir = ic_mean / ic_std if ic_mean is not None and ic_std and ic_std != 0 else None
                    self.ic = ic
                    self.ir = ir
                    self.log_records_summary = {
                        'ic': ic,
                        'ir': ir,
                        'log_records': self.log_records,
                        'num_trades': self.trade_count,
                        'win_rate': (self.win_count / self.trade_count) if self.trade_count > 0 else None
                    }
                    # 可视化回测结果
                    try:
                        plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../alpha_agent_pool/data'))
                        if not os.path.exists(plot_dir):
                            os.makedirs(plot_dir)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(self.equity_curve, label='Equity Curve')
                        ax.set_title('RL Backtest Equity Curve')
                        ax.set_xlabel('Bar Index')
                        ax.set_ylabel('Portfolio Value')
                        ax.legend()
                        plot_path = os.path.join(plot_dir, f"comprehensive_rl_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                        fig.savefig(plot_path)
                        plt.close(fig)
                        self.plot_path = plot_path
                        logger.info(f"RL backtest plot saved: {plot_path}")
                    except Exception as e:
                        logger.warning(f"Failed to plot RL backtest equity curve: {e}")
            cerebro = bt.Cerebro()
            cerebro.broker.setcash(initial_cash)
            cerebro.adddata(data)
            cerebro.addstrategy(StrategyFlowBacktest, signal_data=signals)
            result = cerebro.run()
            strat = result[0]
            ic = getattr(strat, 'ic', None)
            ir = getattr(strat, 'ir', None)
            log_records = getattr(strat, 'log_records', [])
            log_summary = getattr(strat, 'log_records_summary', {})
            plot_path = getattr(strat, 'plot_path', None)
            # 保存完整策略流和回测历史
            history_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../alpha_agent_pool/data'))
            if not os.path.exists(history_dir):
                os.makedirs(history_dir)
            strategy_flow_save = os.path.join(history_dir, f"strategy_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            backtest_history_save = os.path.join(history_dir, f"backtest_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            try:
                with open(strategy_flow_save, 'w') as f:
                    json.dump(flow, f, indent=2)
                with open(backtest_history_save, 'w') as f:
                    json.dump(log_records, f, indent=2)
                logger.info(f"Saved strategy flow: {strategy_flow_save}")
                logger.info(f"Saved backtest history: {backtest_history_save}")
            except Exception as e:
                logger.warning(f"Failed to save strategy flow or backtest history: {e}")
            # 2. RL update
            backtest_results = {
                "IC": ic,
                "IR": ir,
                "log_records": log_records,
                "num_trades": log_summary.get('num_trades', 0),
                "win_rate": log_summary.get('win_rate', None),
                "plot_path": plot_path,
                "strategy_flow_path": strategy_flow_save,
                "backtest_history_path": backtest_history_save
            }
            self.learn_from_backtest(backtest_results)
            if rl_method == "q_learning":
                self.rl_update_policy(learning_rate=learning_rate)
            elif rl_method == "policy_gradient":
                self.rl_policy_gradient_update(baseline=baseline, learning_rate=learning_rate)
            logger.info(f"[RL] RL backtest results: IC={ic}, IR={ir}, log_records={len(log_records)}")
            logger.info(f"[RL] RL update complete. Current window: {getattr(self.config.strategy, 'window', None)}")
            return backtest_results
        except Exception as e:
            logger.error(f"RL backtest and update failed: {e}")
            return None
    def rl_update_policy(self, learning_rate: float = 0.1):
        """
        Advanced RL-style: Q-learning update for window selection.
        Each window is treated as an action, and Q-values are updated based on observed returns.
        This enables the agent to learn a policy for selecting the optimal momentum window over time.
        """
        # Initialize Q-table if not present
        if not hasattr(self, 'window_q_table'):
            self.window_q_table = {}

        # Use feedback history to update Q-values
        for feedback in self.feedback_history[-10:]:  # Use last 10 feedbacks for stability
            window_stats = feedback.get('window_stats', {})
            for w, stats in window_stats.items():
                reward = stats['avg_return']
                old_q = self.window_q_table.get(w, 0.0)
                # Q-learning update: Q(s,a) = Q(s,a) + lr * (reward - Q(s,a))
                new_q = old_q + learning_rate * (reward - old_q)
                self.window_q_table[w] = new_q

        # Select window with highest Q-value
        if self.window_q_table:
            best_q_window = max(self.window_q_table.keys(), key=lambda w: self.window_q_table[w])
            if hasattr(self.config.strategy, 'window'):
                self.config.strategy.window = best_q_window
                logger.info(f"[RL-Q] Updated agent window to {best_q_window} using Q-learning policy.")

    def rl_policy_gradient_update(self, baseline: float = 0.0, learning_rate: float = 0.05):
        """
        Advanced RL-style: Policy gradient update for window selection.
        Each window is assigned a probability, updated based on its advantage (return - baseline).
        """
        # Initialize policy if not present
        if not hasattr(self, 'window_policy'):
            self.window_policy = {}

        # Collect returns for each window
        window_returns = {}
        for feedback in self.feedback_history[-10:]:
            window_stats = feedback.get('window_stats', {})
            for w, stats in window_stats.items():
                window_returns.setdefault(w, []).append(stats['avg_return'])

        # Compute mean returns and update probabilities
        total = 0.0
        for w, returns in window_returns.items():
            mean_ret = sum(returns) / len(returns) if returns else 0.0
            advantage = mean_ret - baseline
            old_prob = self.window_policy.get(w, 1.0)
            new_prob = max(0.01, old_prob + learning_rate * advantage)
            self.window_policy[w] = new_prob
            total += new_prob

        # Normalize probabilities
        for w in self.window_policy:
            self.window_policy[w] /= total if total > 0 else 1.0

        # Sample window according to policy (softmax)
        import random
        windows, probs = zip(*self.window_policy.items())
        chosen_window = random.choices(windows, weights=probs, k=1)[0]
        if hasattr(self.config.strategy, 'window'):
            self.config.strategy.window = chosen_window
            logger.info(f"[RL-PG] Updated agent window to {chosen_window} using policy gradient.")
    def __init__(self, config: MomentumAgentConfig):
        """
        Initialize the MomentumAgent with A2A protocol integration for memory communication.
        
        Args:
            config (MomentumAgentConfig): Configuration object for the agent.
        """
        self.config = config
        self.agent = FastMCP("MomentumAlphaAgent")
        
        # Initialize A2A client for communication with memory agent through alpha pool
        self.a2a_client = create_alpha_pool_a2a_client(
            agent_pool_id="alpha_agent_pool",
            memory_url="http://127.0.0.1:8010"
        )
        
        # Initialize LLM client
        self.llm_client = None
        self._initialize_llm()
        
        # Path to store strategy signal flow (local fallback)
        self.signal_flow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../momentum_signal_flow.json"))
        
        # Clear signal flow file on agent restart
        if os.path.exists(self.signal_flow_path):
            try:
                os.remove(self.signal_flow_path)
            except Exception:
                pass
                    
        self._register_tools()

        # Feedback history for RL-style learning (now stored in A2A memory)
        self.feedback_history = []
        
        logger.info("MomentumAgent initialized with A2A memory integration")

    async def learn_from_backtest(self, backtest_results: dict):
        """
        Learn from backtest results and store insights using A2A protocol.
        Args:
            backtest_results (dict): Detailed per-bar logs and factor metrics (IC/IR, etc.)
        """
        import os
        # Extract key metrics from backtest results
        log_records = backtest_results.get("log_records", [])
        ic = backtest_results.get("IC", None)
        ir = backtest_results.get("IR", None)
        signal_stats = backtest_results.get("signal_stats", {})

        # Analyze window performance for strategy adaptation
        window_performance = {}
        for r in log_records:
            w = r.get("selected_timeframe")
            ret = r.get("daily_return", 0)
            if w is not None:
                if w not in window_performance:
                    window_performance[w] = {"returns": [], "count": 0}
                window_performance[w]["returns"].append(ret)
                window_performance[w]["count"] += 1

        # Calculate average returns for each window
        window_stats = {w: {
            "avg_return": (sum(d["returns"]) / len(d["returns"]) if d["returns"] else 0),
            "count": d["count"]
        } for w, d in window_performance.items()}

        # Select best performing window for strategy optimization
        best_window = None
        if window_stats:
            best_window = max(window_stats.keys(), key=lambda w: window_stats[w]["avg_return"])

        # Prepare performance metrics for storage
        performance_metrics = {
            "IC": ic,
            "IR": ir,
            "signal_stats": signal_stats,
            "num_trades": len([r for r in log_records if r.get("trade_executed", False)]),
            "win_rate": signal_stats.get("win_rate", None),
            "avg_return": signal_stats.get("avg_return", None),
            "momentum_windows": list(set(r.get("selected_timeframe") for r in log_records if "selected_timeframe" in r)),
            "window_stats": window_stats,
            "best_window": best_window,
            "feedback_time": datetime.utcnow().isoformat() + "Z",
            "total_log_records": len(log_records)
        }

        # Auto-update agent window based on performance
        if best_window is not None and hasattr(self.config.strategy, 'window'):
            old_window = getattr(self.config.strategy, 'window', None)
            self.config.strategy.window = best_window
            logger.info(f"[RL] Auto-updated agent window from {old_window} to {best_window} based on backtest feedback.")

        # Store performance results using A2A protocol
        try:
            async with self.a2a_client as client:
                # Store strategy performance metrics
                strategy_id = f"momentum_strategy_{self.config.agent_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                await client.store_strategy_performance(
                    agent_id=self.config.agent_id,
                    strategy_id=strategy_id,
                    performance_metrics=performance_metrics
                )
                
                # Store learning feedback for RL adaptation
                learning_feedback = {
                    "window_adaptation": {
                        "old_window": old_window,
                        "new_window": best_window,
                        "window_performance": window_stats
                    },
                    "strategy_metrics": performance_metrics,
                    "adaptation_timestamp": datetime.utcnow().isoformat()
                }
                
                await client.store_learning_feedback(
                    agent_id=self.config.agent_id,
                    feedback_type="MOMENTUM_WINDOW_ADAPTATION",
                    feedback_data=learning_feedback
                )
                
                logger.info(f"[A2A] Successfully stored backtest results and learning feedback via A2A protocol")
                
        except Exception as e:
            logger.warning(f"[A2A] Failed to store results via A2A protocol: {e}")
            
            # Fallback to local storage
            self.feedback_history.append(performance_metrics)
            try:
                feedback_path = os.path.join(os.path.dirname(__file__), "../../momentum_agent_feedback.json")
                with open(feedback_path, "w") as f:
                    json.dump(self.feedback_history, f, indent=2)
                logger.info(f"[FALLBACK] Stored feedback locally at {feedback_path}")
            except Exception as local_e:
                logger.error(f"[FALLBACK] Failed to store feedback locally: {local_e}")

        # Additional local data backup for analysis
        try:
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            # Save detailed feedback with timestamp
            feedback_save = os.path.join(data_dir, f"momentum_agent_feedback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
            with open(feedback_save, "w") as f:
                json.dump(performance_metrics, f, indent=2)
            
            # Save strategy flow if available
            if hasattr(self, 'signal_flow_path') and os.path.exists(self.signal_flow_path):
                import shutil
                flow_save = os.path.join(data_dir, f"strategy_flow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
                shutil.copy(self.signal_flow_path, flow_save)
                logger.info(f"[BACKUP] Saved strategy flow to {flow_save}")
            
            logger.info(f"[BACKUP] Saved detailed feedback to {feedback_save}")
            
        except Exception as e:
            logger.warning(f"[BACKUP] Failed to save backup data: {e}")

        logger.info(f"[LEARN] Feedback processing completed: IC={ic}, IR={ir}, win_rate={performance_metrics.get('win_rate')}")
        return performance_metrics
    
    async def _store_signal_via_a2a(self, 
                                   symbol: str, 
                                   signal: str, 
                                   confidence: float, 
                                   reasoning: str, 
                                   market_context: Dict[str, Any], 
                                   request_id: str):
        """
        Store trading signal event via A2A protocol (non-blocking).
        
        Args:
            symbol: Trading symbol
            signal: Trading signal (BUY/SELL/HOLD)
            confidence: Signal confidence score
            reasoning: Signal reasoning
            market_context: Market context data
            request_id: Request identifier for correlation
        """
        try:
            async with self.a2a_client as client:
                await client.store_alpha_signal_event(
                    agent_id=self.config.agent_id,
                    signal=signal,
                    confidence=confidence,
                    symbol=symbol,
                    reasoning=reasoning,
                    market_context=market_context,
                    correlation_id=request_id
                )
                logger.info(f"[A2A] Successfully stored signal event for {symbol} via A2A protocol")
        except Exception as e:
            logger.warning(f"[A2A] Failed to store signal event via A2A protocol: {e}")
            # Continue execution without breaking the main signal generation flow

    def _initialize_llm(self):
        """Initialize the LLM client for intelligent analysis."""
        logger.info("[DEBUG] Entering _initialize_llm()")
        if not LLM_AVAILABLE:
            logger.warning("LLM dependencies not available")
            return
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            logger.info(f"[DEBUG] OPENAI_API_KEY loaded: {'YES' if api_key else 'NO'}")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment")
                return
            self.llm_client = AsyncOpenAI(api_key=api_key)
            logger.info("✅ LLM client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}", exc_info=True)

    def _analyze_multiple_timeframes(self, prices: List[float]) -> Dict[str, Any]:
        """Analyze momentum across multiple timeframes to make intelligent decisions."""
        if len(prices) < 5:
            # For insufficient data, provide a basic fallback structure
            basic_momentum = self._calculate_momentum(prices, min(len(prices), 2)) if len(prices) >= 2 else 0.0
            basic_volatility = self._calculate_volatility(prices) if len(prices) >= 2 else 0.1
            return {
                "analysis": {"insufficient_data": True},
                "windows": [],
                "best_window": min(5, len(prices)),
                "selected_momentum": basic_momentum,
                "selected_volatility": basic_volatility,
                "windows_tested": [min(5, len(prices))],
                "adaptive_selection": False
            }
        
        # Test multiple lookback windows adaptively based on data length
        max_data = len(prices)
        windows = [w for w in [5, 10, 15, 20, 30, 50] if w <= max_data]
        if not windows:
            windows = [min(5, max_data)]
        
        window_analysis = {}
        
        for window in windows:
            if len(prices) >= window:
                momentum = self._calculate_momentum(prices, window)
                volatility = self._calculate_volatility(prices[-window:])
                
                # Calculate trend strength and consistency
                trend_strength = abs(momentum) / (volatility + 1e-8)
                
                # Calculate trend consistency across the window
                window_prices = prices[-window:]
                short_term_momentum = self._calculate_momentum(window_prices, min(5, len(window_prices)))
                consistency = 1.0 - abs(momentum - short_term_momentum) / (abs(momentum) + 1e-8)
                
                # Signal quality combines strength, low volatility, and consistency
                signal_quality = trend_strength * (1 - min(volatility, 0.5)) * consistency
                
                window_analysis[window] = {
                    "momentum": momentum,
                    "volatility": volatility,
                    "trend_strength": trend_strength,
                    "consistency": consistency,
                    "signal_quality": signal_quality
                }
        
        # Select best window based on signal quality
        if window_analysis:
            best_window = max(window_analysis.keys(), key=lambda w: window_analysis[w]["signal_quality"])
        else:
            best_window = 5
        
        return {
            "analysis": window_analysis,
            "best_window": best_window,
            "selected_momentum": window_analysis.get(best_window, {}).get("momentum", 0.0),
            "selected_volatility": window_analysis.get(best_window, {}).get("volatility", 0.1),
            "windows_tested": list(windows),
            "adaptive_selection": True
        }

    async def _analyze_market_with_llm(self, symbol: str, prices: List[float], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to analyze market conditions with intelligent multi-timeframe analysis and RL-style feedback integration.
        RL-style learning: inject recent backtest feedback into the prompt and adapt agent parameters.
        """
        logger.info(f"[DEBUG] Entering _analyze_market_with_llm for symbol={symbol}, num_prices={len(prices)}")
        if not self.llm_client:
            logger.warning("LLM client not initialized, using fallback analysis.")
            return self._fallback_analysis(symbol, prices, context)

        # Step 1: RL-style feedback integration
        feedback_summary = self.feedback_history[-1] if self.feedback_history else None
        feedback_text = ""
        if feedback_summary:
            feedback_text = (
                f"\nRECENT BACKTEST FEEDBACK:\n"
                f"- Information Coefficient (IC): {feedback_summary.get('IC', 'N/A')}\n"
                f"- Information Ratio (IR): {feedback_summary.get('IR', 'N/A')}\n"
                f"- Win Rate: {feedback_summary.get('win_rate', 'N/A')}\n"
                f"- Average Return: {feedback_summary.get('avg_return', 'N/A')}\n"
                f"- Momentum Windows Used: {feedback_summary.get('momentum_windows', [])}\n"
                f"- Feedback Time: {feedback_summary.get('feedback_time', 'N/A')}\n"
                f"- Signal Stats: {json.dumps(feedback_summary.get('signal_stats', {}), indent=2)}\n"
                f"\nPlease adapt your momentum factor mining and signal logic to maximize IC/IR and win rate, and avoid windows or factors that performed poorly."
            )

        # Step 2: RL-style parameter adaptation
        if feedback_summary and feedback_summary.get('momentum_windows'):
            preferred_windows = feedback_summary['momentum_windows']
            if hasattr(self.config.strategy, 'window') and preferred_windows:
                self.config.strategy.window = preferred_windows[-1]
                logger.info(f"[RL] Updated preferred momentum window to {self.config.strategy.window} based on feedback.")

        # Step 3: Multi-timeframe analysis (re-run after possible window update)
        timeframe_analysis = self._analyze_multiple_timeframes(prices)
        best_window = timeframe_analysis["best_window"]
        if len(prices) >= 2:
            current_price = prices[-1]
            price_change_5d = (prices[-1] - prices[-min(5, len(prices))]) / prices[-min(5, len(prices))] * 100
            price_change_total = (prices[-1] - prices[0]) / prices[0] * 100
            volatility = timeframe_analysis["selected_volatility"]
            momentum = timeframe_analysis["selected_momentum"]
        else:
            current_price = prices[0] if prices else 0
            price_change_5d = 0
            price_change_total = 0
            volatility = 0
            momentum = 0

        # Step 4: Construct LLM prompt with feedback
        prompt = f"""
Analyze market data for {symbol} using intelligent multi-timeframe momentum analysis:

CURRENT MARKET STATE:
- Current Price: ${current_price:.2f}
- Recent Price Data: {prices[-10:] if len(prices) > 10 else prices}

MULTI-TIMEFRAME ANALYSIS RESULTS:
- Optimal Window Selected: {best_window} periods (from candidates: {timeframe_analysis['windows_tested']})
- Selected Momentum: {momentum:.4f} ({momentum*100:.2f}%)
- Price Change (5d): {price_change_5d:.2f}%
- Total Price Change: {price_change_total:.2f}%
- Volatility: {volatility:.4f}
- Signal Quality Score: {timeframe_analysis['analysis'].get(best_window, {}).get('signal_quality', 0):.4f}

DETAILED WINDOW COMPARISON:
{json.dumps(timeframe_analysis['analysis'], indent=2)}

MARKET CONTEXT: {context}
{feedback_text}

INTELLIGENT TRADING DECISION REQUIRED:
As an expert quantitative analyst, provide an adaptive trading signal considering:

1. MULTI-TIMEFRAME CONVERGENCE: How do different timeframes align?
2. SIGNAL QUALITY: Is the trend strong and consistent across the optimal window?
3. RISK-ADJUSTED RETURNS: What's the expected return vs. risk?
4. MARKET REGIME ADAPTATION: What type of market environment are we in?
5. EXECUTION SIZING: What position size is appropriate given confidence?
6. RL FEEDBACK: Use the above backtest feedback to improve your momentum factor mining and signal logic.

Return ONLY valid JSON with these fields:
{{
  "signal": "BUY|SELL|HOLD",
  "confidence": 0.0-1.0,
  "reasoning": "detailed explanation of multi-timeframe analysis and decision logic",
  "market_regime": "bullish_trend|bearish_trend|neutral|volatile|trending",
  "predicted_return": expected_return_estimate,
  "risk_estimate": 0.0-1.0,
  "key_factors": ["list", "of", "key", "decision", "factors"],
  "selected_timeframe": {best_window},
  "execution_weight": 0.0-1.0
}}

Focus on intelligent adaptation and RL-style learning. Use the multi-timeframe analysis and feedback to make nuanced decisions.
"""

        logger.info(f"[LLM DEBUG] Selected timeframe: {best_window}, momentum: {momentum:.4f}")
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": "You are an expert quantitative analyst specializing in multi-timeframe momentum analysis. Make intelligent, adaptive trading decisions."},
                    {"role": "user", "content": prompt}
                ],
            )

            result_text = response.choices[0].message.content
            logger.info(f"[LLM DEBUG] Raw model output: {result_text}")

            # Parse JSON response
            try:
                llm_analysis = json.loads(result_text)
                logger.info(f"[LLM DEBUG] Parsed LLM JSON: {llm_analysis}")
                
                # Add timeframe analysis metadata
                llm_analysis.update({
                    "selected_timeframe": best_window,
                    "timeframe_analysis": timeframe_analysis,
                    "execution_weight": abs(float(llm_analysis.get("confidence", 0.5))),
                    "analysis_source": "llm_multiframe"
                })
                
            except json.JSONDecodeError:
                logger.error(f"LLM raw output cannot be parsed: {result_text}")
                return self._fallback_analysis(symbol, prices, context)

            logger.info(f"[LLM DEBUG] Final intelligent analysis result: {llm_analysis}")
            return llm_analysis

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}", exc_info=True)
            return self._fallback_analysis(symbol, prices, context)

    def _fallback_analysis(self, symbol: str, prices: List[float], context: Dict[str, Any]) -> Dict[str, Any]:
        """Direct momentum analysis without conservative thresholds - let the signal speak for itself."""
        if len(prices) < 2:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": "Insufficient price data for analysis",
                "market_regime": "neutral",
                "predicted_return": 0.0,
                "risk_estimate": 0.1,
                "key_factors": ["insufficient_data"],
                "execution_weight": 0.0,
                "analysis_source": "fallback",
                "selected_timeframe": 5
            }
        
        # Use intelligent timeframe analysis
        timeframe_analysis = self._analyze_multiple_timeframes(prices)
        best_window = timeframe_analysis["best_window"]
        momentum = timeframe_analysis["selected_momentum"]
        volatility = timeframe_analysis["selected_volatility"]
        
        # DIRECT SIGNAL GENERATION - NO CONSERVATIVE THRESHOLDS
        signal = "HOLD"
        confidence = 0.0
        
        # Get trend quality safely
        if isinstance(timeframe_analysis["analysis"], dict) and best_window in timeframe_analysis["analysis"]:
            trend_quality = timeframe_analysis["analysis"][best_window]["signal_quality"]
        else:
            trend_quality = abs(momentum) / (volatility + 1e-8) * 0.5
        
        # DIRECT MOMENTUM-BASED SIGNALS (no arbitrary thresholds)
        if momentum > 0.005:  # Minimal threshold - 0.5% to filter noise only
            signal = "BUY"
            confidence = min(0.95, max(0.3, trend_quality * 3))  # Scale up confidence
        elif momentum < -0.005:  # Minimal threshold - 0.5% to filter noise only  
            signal = "SELL"
            confidence = min(0.95, max(0.3, trend_quality * 3))  # Scale up confidence
        else:
            signal = "HOLD"
            confidence = max(0.2, min(0.6, trend_quality * 2))  # Base confidence for HOLD
        
        # Market regime based on momentum direction
        if momentum > 0.002:
            regime = "bullish_trend"
        elif momentum < -0.002:
            regime = "bearish_trend" 
        elif volatility > 0.04:
            regime = "volatile"
        else:
            regime = "neutral"
        
        # Build key factors
        key_factors = [
            f"timeframe_{best_window}",
            f"momentum_{momentum:.4f}",
            f"volatility_{volatility:.4f}",
            f"quality_{trend_quality:.4f}",
            "direct_signal_no_smoothing"
        ]
        
        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": f"Direct momentum signal (window={best_window}): momentum={momentum:.4f}, quality={trend_quality:.4f} - No conservative smoothing applied",
            "market_regime": regime,
            "predicted_return": momentum * 1.0,  # Direct momentum scaling
            "risk_estimate": min(0.9, volatility * 2),
            "key_factors": key_factors,
            "execution_weight": confidence,  # Use full confidence as execution weight
            "analysis_source": "fallback_direct",
            "selected_timeframe": best_window,
            "timeframe_analysis": timeframe_analysis
        }

    def _calculate_momentum(self, prices: List[float], window: Optional[int] = None) -> float:
        """Calculate momentum indicator with intelligent window selection."""
        if window is None:
            window = self.config.strategy.window
            
        if len(prices) < window:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-window]
        return (current_price - past_price) / past_price if past_price != 0 else 0.0

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility."""
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] != 0]
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    def _read_memory(self, key: str):
        """Read a value from the shared memory unit."""
        if not os.path.exists(self.memory_path):
            return None
        try:
            with open(self.memory_path, 'r') as f:
                memory = json.load(f)
            return memory.get(key)
        except Exception as e:
            logger.warning(f"Failed to read memory: {e}")
            return None

    def _write_signal_flow(self, flow_data: dict):
        """Write strategy signal flow to a JSON file."""
        try:
            with open(self.signal_flow_path, 'w') as f:
                json.dump(flow_data, f, indent=2)
            logger.info(f"Signal flow written to {self.signal_flow_path}")
        except Exception as e:
            logger.warning(f"Failed to write signal flow: {e}")

    def _register_tools(self):
        """
        Register the agent's tools with the FastMCP server with LLM-powered analysis.
        """
        @self.agent.tool()
        async def generate_signal(symbol: str, price_list: Optional[List[float]] = None, ctx: MCPContext = None) -> dict:
            """
            Generate a sophisticated alpha strategy flow using LLM analysis and real market data.
            """
            request_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"[REQUEST {request_id}] generate_signal called with symbol={symbol}, price_list_length={len(price_list) if price_list else 0}")
            
            try:
                # Convert parameters to MomentumSignalRequest object for internal processing
                try:
                    request = MomentumSignalRequest(symbol=symbol, price_list=price_list)
                    logger.info(f"[REQUEST {request_id}] Successfully created MomentumSignalRequest")
                except Exception as e:
                    logger.error(f"[REQUEST {request_id}] Error creating MomentumSignalRequest: {e}", exc_info=True)
                    return {
                        "signal": "HOLD",
                        "confidence": 0.0,
                        "error": f"Invalid request parameters: {e}",
                        "reasoning": f"Failed to parse request: {e}",
                        "request_id": request_id
                    }

                symbol = request.symbol
                price_list = request.price_list

                # Get price data
                logger.info(f"[REQUEST {request_id}] Getting price data for {symbol}")
                if price_list is None:
                    closes = []
                    for i in range(self.config.strategy.window):
                        key = f"{symbol}_close_2024-01-{str(31-i).zfill(2)} 05:00:00"
                        val = self._read_memory(key)
                        if val is not None:
                            closes.insert(0, float(val))
                    prices = closes if closes else self._generate_synthetic_prices(symbol, self.config.strategy.window)
                    logger.info(f"[REQUEST {request_id}] Using {'memory' if closes else 'synthetic'} price data: {len(prices)} points")
                else:
                    prices = price_list
                    logger.info(f"[REQUEST {request_id}] Using provided price data: {len(prices)} points")

                # Use LLM for intelligent analysis
                market_context_data = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "price_count": len(prices)
                }

                logger.info(f"[REQUEST {request_id}] Starting LLM analysis for {symbol}")
                analysis_result = await self._analyze_market_with_llm(symbol, prices, market_context_data)
                logger.info(f"[REQUEST {request_id}] LLM analysis result: {analysis_result}")

                # Build comprehensive output using LLM insights
                now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
                code_hash = hashlib.sha256((str(prices) + analysis_result["signal"]).encode()).hexdigest()

                # Determine market regime
                regime_map = {
                    "BUY": analysis_result.get("market_regime", "bullish_trend"),
                    "SELL": analysis_result.get("market_regime", "bearish_trend"),
                    "HOLD": analysis_result.get("market_regime", "neutral")
                }

                flow_obj = AlphaStrategyFlow(
                    alpha_id="momentum_llm_v4",
                    version="2025.06.29-llm",
                    timestamp=now,
                    market_context=MarketContext(
                        symbol=symbol,
                        regime_tag=regime_map[analysis_result["signal"]],
                        input_features={
                            "price_current": prices[-1] if prices else 0,
                            "price_20d_ago": prices[0] if len(prices) >= 20 else (prices[0] if prices else 0),
                            "sma_10": sum(prices[-10:]) / min(10, len(prices)) if prices else 0,
                            "sma_20": sum(prices[-20:]) / min(20, len(prices)) if prices else 0,
                            "momentum_score": self._calculate_momentum(prices),
                            "volatility": self._calculate_volatility(prices),
                            "analysis_source": analysis_result.get("analysis_source", "unknown"),
                            "llm_factors": analysis_result.get("key_factors", [])
                        }
                    ),
                    decision=Decision(
                        signal=analysis_result["signal"],
                        confidence=analysis_result["confidence"],
                        reasoning=analysis_result["reasoning"],
                        predicted_return=analysis_result["predicted_return"],
                        risk_estimate=analysis_result["risk_estimate"],
                        signal_type="directional",
                        asset_scope=[symbol]
                    ),
                    # action=Action(
                    #     execution_weight=analysis_result["execution_weight"],
                    #     order_type="market",
                    #     order_price=prices[-1] if prices else 0,
                    #     execution_delay="T+0"
                    # ),
                    performance_feedback=PerformanceFeedback(
                        status="pending",
                        evaluation_link=None
                    ),
                    metadata=Metadata(
                        generator_agent="momentum_llm_agent",
                        strategy_prompt="LLM-powered momentum analysis with sophisticated market regime detection",
                        code_hash=f"sha256:{code_hash}",
                        context_id=f"llm_dag_{now[:10].replace('-', '')}_{now[11:13]}"
                    )
                )

                # Write to strategy flow file, always use the best window (alpha factor) if available
                try:
                    # 动态替换 input_features 里的 window/factor 为最佳 window
                    best_window = None
                    if hasattr(self.config.strategy, 'window'):
                        best_window = self.config.strategy.window
                    # 如果 analysis_result 里有 best_window，优先用
                    if 'best_window' in analysis_result:
                        best_window = analysis_result['best_window']
                    # 更新 input_features 里的 window（如果有）
                    if best_window is not None:
                        flow_obj.market_context.input_features['selected_timeframe'] = best_window
                    # Use model_dump() for newer Pydantic versions, fallback to dict() for older versions
                    if hasattr(flow_obj, 'model_dump'):
                        flow_dict = flow_obj.model_dump()
                    else:
                        flow_dict = flow_obj.dict()
                    self._write_signal_flow(flow_dict)
                except Exception as e:
                    logger.warning(f"Failed to write signal flow: {e}")

                try:
                    # Use model_dump() for newer Pydantic versions, fallback to dict() for older versions
                    if hasattr(flow_obj, 'model_dump'):
                        result = flow_obj.model_dump()
                    else:
                        result = flow_obj.dict()
                    
                    # Store signal event using A2A protocol
                    asyncio.create_task(self._store_signal_via_a2a(
                        symbol=symbol,
                        signal=analysis_result["signal"],
                        confidence=analysis_result["confidence"],
                        reasoning=analysis_result["reasoning"],
                        market_context=market_context_data,
                        request_id=request_id
                    ))
                    
                    logger.info(f"[REQUEST {request_id}] Successfully generated signal for {symbol}: {result.get('decision', {}).get('signal', 'UNKNOWN')}")
                    return result
                except Exception as e:
                    logger.error(f"[REQUEST {request_id}] Error converting flow object to dict: {e}", exc_info=True)
                    # Return a simplified response if the full object fails
                    return {
                        "signal": analysis_result.get("signal", "HOLD"),
                        "confidence": analysis_result.get("confidence", 0.0),
                        "reasoning": analysis_result.get("reasoning", "Error converting response"),
                        "error": str(e),
                        "request_id": request_id
                    }
            except Exception as e:
                logger.error(f"[REQUEST {request_id}] Error executing tool generate_signal: {e}", exc_info=True)
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "raw_response": f"Error executing tool generate_signal: {e}",
                    "request_id": request_id
                }

        @self.agent.tool()
        async def analyze_market_sentiment(symbol: str, lookback_days: int = 20) -> dict:
            """
            Analyze market sentiment using LLM for the given symbol.
            """
            try:
                # Get price data
                prices = []
                for i in range(lookback_days):
                    key = f"{symbol}_close_2024-01-{str(31-i).zfill(2)} 05:00:00"
                    val = self._read_memory(key)
                    if val is not None:
                        prices.insert(0, float(val))
                
                if not prices:
                    prices = self._generate_synthetic_prices(symbol, lookback_days)
                
                # Use LLM for sentiment analysis
                analysis = await self._analyze_market_with_llm(symbol, prices, {})
                
                return {
                    "symbol": symbol,
                    "sentiment": analysis.get("market_regime", "neutral"),
                    "confidence": analysis.get("confidence", 0.0),
                    "key_factors": analysis.get("key_factors", []),
                    "reasoning": analysis.get("reasoning", ""),
                    "timestamp": datetime.now().isoformat(),
                    "analysis_source": analysis.get("analysis_source", "llm")
                }
                
            except Exception as e:
                return {
                    "symbol": symbol,
                    "sentiment": "unknown",
                    "confidence": 0.0,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        @self.agent.tool()
        async def run_rl_backtest_and_update(symbol: str, market_data: list, lookback_period: int = 30, initial_cash: float = 100000.0) -> dict:
            """
            Run RL backtest and update agent policy for a given symbol and market data.
            All signals are generated by LLM or direct momentum analysis, no hardcoded logic.
            Args:
                symbol (str): The symbol to backtest.
                market_data (list): List of dicts with 'date' and 'price'.
                lookback_period (int): Lookback window for momentum.
                initial_cash (float): Initial cash for backtest.
            Returns:
                dict: RL backtest and update results.
            """
            import pandas as pd
            import tempfile
            import json
            try:
                # Prepare market data CSV
                df = pd.DataFrame(market_data)
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as f:
                    df.to_csv(f.name, index=False)
                    market_data_path = f.name
                # Generate signals using LLM or direct momentum analysis
                prices = [entry['price'] for entry in market_data]
                dummy_flow = []
                # For each bar, use all available prices up到当前bar
                for i, entry in enumerate(market_data):
                    price_list = prices[:i+1]
                    # Prefer LLM if available, fallback to direct momentum
                    if self.llm_client:
                        analysis_result = await self._analyze_market_with_llm(symbol, price_list, {"date": entry.get("date", "")})
                    else:
                        analysis_result = self._fallback_analysis(symbol, price_list, {"date": entry.get("date", "")})
                    dummy_flow.append({
                        "alpha_signals": {
                            "signals": {
                                symbol: {
                                    "decision": {
                                        "signal": analysis_result.get("signal", "HOLD"),
                                        "confidence": analysis_result.get("confidence", 0.0),
                                        "predicted_return": analysis_result.get("predicted_return", 0.0),
                                        "reasoning": analysis_result.get("reasoning", "")
                                    },
                                    "action": {
                                        "execution_weight": analysis_result.get("execution_weight", 0.0)
                                    },
                                    "selected_timeframe": analysis_result.get("selected_timeframe", lookback_period)
                                }
                            }
                        }
                    })
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
                    json.dump(dummy_flow, f)
                    strategy_flow_path = f.name
                # Call internal RL backtest and update
                results = self.run_rl_backtest_and_update(strategy_flow_path, market_data_path, rl_method="q_learning", learning_rate=0.1, baseline=0.0, feedback_days=10, initial_cash=initial_cash)
                return {"status": "success", "results": results}
            except Exception as e:
                logger.error(f"Error in run_rl_backtest_and_update tool: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {"status": "error", "message": str(e)}

    def _generate_synthetic_prices(self, symbol: str, window: int) -> List[float]:
        """
        Generate realistic synthetic price series for testing when real data is unavailable.
        """
        import random
        base_price = 100.0
        prices = [base_price]
        
        for _ in range(window - 1):
            # Simple random walk with slight upward bias
            change = random.gauss(0.001, 0.02)  # 0.1% daily drift, 2% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Ensure price stays positive
        
        return prices

    def start_mcp_server(self, host="0.0.0.0", port: Optional[int] = None, use_sse: bool = True):
        """
        Start the MCP server for the agent, supporting SSE communication by default.
        Args:
            host (str): The host to bind the server to.
            port (int, optional): The port to run the server on. Defaults to config.
            use_sse (bool): Whether to use SSE (Server-Sent Events) for communication. Default is True.
        """
        port = port or self.config.execution.port
        self.agent.settings.host = host
        self.agent.settings.port = port
        
        # Choose transport based on use_sse parameter
        transport = "sse" if use_sse else "stdio"
        
        logger.info(f"Starting LLM-powered MomentumAgent MCP server on {host}:{port} with transport: {transport}")
        if self.llm_client:
            logger.info("🤖 LLM integration enabled - using intelligent analysis")
        else:
            logger.warning("⚠️ LLM not available - using fallback analysis")

        self.agent.run(transport=transport)

    def __repr__(self):
        """
        Return a string representation of the enhanced MomentumAgent.
        """
        llm_status = "LLM-enabled" if self.llm_client else "fallback-mode"
        return f"<MomentumAgent id={self.config.agent_id} port={self.config.execution.port} {llm_status}>"


def main(config_dict=None):
    import os
    import sys
    import yaml
    
    def to_dict_recursive(obj):
        """
        Recursively convert an object to a dict, compatible with pydantic, dataclass, and normal objects.
        """
        if isinstance(obj, dict):
            return {k: to_dict_recursive(v) for k, v in obj.items()}
        if hasattr(obj, 'model_dump'):
            return to_dict_recursive(obj.model_dump())
        if hasattr(obj, '__dict__') and not isinstance(obj, type):
            return to_dict_recursive(vars(obj))
        if isinstance(obj, (list, tuple, set)):
            return [to_dict_recursive(i) for i in obj]
        return obj

    if config_dict is not None:
        # If config_dict is provided, use it to initialize the agent and start the MCP server.
        config_dict = to_dict_recursive(config_dict)
        config = MomentumAgentConfig(**config_dict)
        agent = MomentumAgent(config)
        agent.start_mcp_server()
        return
    
    import argparse
    parser = argparse.ArgumentParser(description="Start MomentumAgent MCP server.")
    parser.add_argument('--sse', action='store_true', help='Use SSE channel (default, reserved for future extension)')
    args = parser.parse_args()

    # Load config from YAML file if no config_dict is provided
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/momentum.yaml"))
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    config = MomentumAgentConfig(**config_data)
    agent = MomentumAgent(config)
    # Default to SSE channel; parameter reserved for future use
    agent.start_mcp_server()


if __name__ == "__main__":
    main()
