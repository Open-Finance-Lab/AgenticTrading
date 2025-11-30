import sys
import os
import json
from pathlib import Path

# Add parent directory to path to find FinAgents
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from FinAgents.orchestrator_demo.orchestrator import Orchestrator

def train_agents_over_period(symbols, start_year, end_year):
    # Initialize Orchestrator (which initializes Agents with GPT-4o)
    orchestrator = Orchestrator()
    
    current_prompts = {
        "Alpha": orchestrator.alpha_agent.agent.instructions,
        "Risk": orchestrator.risk_agent.agent.instructions,
        "Portfolio": orchestrator.portfolio_agent.agent.instructions
    }
    
    performance_history = []
    
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        print(f"\n--- Processing Year: {year} ---")
        
        try:
            # Run Pipeline with multiple symbols
            # Orchestrator.run_pipeline now accepts a list of symbols
            result = orchestrator.run_pipeline(symbols, start_date, end_date, mode="backtest")
            
            if result and result.get('status') == 'success':
                metrics = result.get('performance_metrics', {})
                sharpe = metrics.get('sharpe_ratio', 0.0)
                total_return = metrics.get('total_return', 0.0)
                print(f"📊 Performance for {year}: Sharpe Ratio = {sharpe:.2f}, Return = {total_return:.2%}")
                
                performance_history.append({'year': year, 'sharpe': sharpe, 'return': total_return})
                
                # Optimization Logic
                if sharpe < 1.0:
                    print("⚠️ Performance below threshold. Optimizing prompts...")
                    
                    # Optimize Alpha Agent
                    new_instruction = orchestrator.optimize_agent_prompts(
                        agent_name="Alpha", 
                        performance_metric="Sharpe Ratio", 
                        current_value=sharpe, 
                        target_value=1.5
                    )
                    
                    if new_instruction and "Optimization failed" not in new_instruction:
                         current_prompts["Alpha"] = new_instruction
                         print("✅ Alpha Agent prompt updated.")
            else:
                print(f"❌ Backtest failed for {year}: {result.get('message') if result else 'Unknown error'}")
                
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()
            
    return current_prompts, performance_history

if __name__ == "__main__":
    # Use a universe of stocks for better portfolio diversification
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
    print(f"🚀 Starting Backtest Training with symbols: {symbols}")
    
    # Run training from 2010 to 2023
    optimized_prompts, history = train_agents_over_period(symbols, 2019, 2023)
    
    # Save optimized prompts
    output_file = os.path.join(current_dir, "optimized_prompts.json")
    with open(output_file, "w") as f:
        json.dump(optimized_prompts, f, indent=2)
    print(f"\n✅ Optimization Complete. Prompts saved to {output_file}")

