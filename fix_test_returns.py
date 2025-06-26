#!/usr/bin/env python3
"""
Script to fix test functions by removing return statements
and adding proper pytest assertions.
"""

import re

def fix_test_file(file_path):
    """Fix test functions to use assertions instead of returns."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace return statements in test functions
    patterns_replacements = [
        # Remove return True/False statements and replace with assertions
        (r'return True', ''),
        (r'return False', 'pytest.fail("Test failed")'),
        
        # Replace return [...] with assertions for list returns
        (r'return predictions', '''# Assertions to validate predictions
    assert len(predictions) > 0, "No predictions generated"
    assert all('predicted_cost_bps' in p for p in predictions), "Missing prediction data"'''),
        
        (r'return impact_estimates', '''# Assertions to validate impact estimates  
    assert len(impact_estimates) > 0, "No impact estimates generated"
    assert all('total_impact_bps' in e for e in impact_estimates), "Missing impact data"'''),
        
        (r'return execution_analyses', '''# Assertions to validate execution analyses
    assert len(execution_analyses) > 0, "No execution analyses generated"
    assert all('implementation_shortfall_bps' in a for a in execution_analyses), "Missing execution data"'''),
        
        (r'return optimization_results', '''# Assertions to validate optimization results
    assert len(optimization_results) > 0, "No optimization results generated"  
    assert all('optimization_status' in r for r in optimization_results), "Missing optimization data"'''),
        
        (r'return risk_analyses', '''# Assertions to validate risk analyses
    assert len(risk_analyses) > 0, "No risk analyses generated"
    assert all('risk_adjusted_cost_bps' in r for r in risk_analyses), "Missing risk data"'''),
        
        (r'return event_ids', '''# Assertions to validate memory operations
    assert len(event_ids) > 0, "No events stored in memory"
    assert all(isinstance(eid, str) for eid in event_ids), "Invalid event IDs"'''),
        
        # Replace try/except blocks that return values
        (r'except Exception as e:\s+print\(f"[^"]*\{e\}"\)\s+return \[\]', 
         'except Exception as e:\n        pytest.fail(f"Test failed with error: {e}")'),
         
        (r'except Exception as e:\s+print\(f"[^"]*\{e\}"\)\s+return False', 
         'except Exception as e:\n        pytest.fail(f"Test failed with error: {e}")'),
    ]
    
    # Apply replacements
    for pattern, replacement in patterns_replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Add pytest import if not present
    if 'import pytest' not in content:
        content = content.replace('import asyncio', 'import asyncio\nimport pytest')
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed test file: {file_path}")

if __name__ == "__main__":
    fix_test_file("/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/tests/test_transaction_cost_pool_comprehensive_english.py")
