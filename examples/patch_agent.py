import os

def patch_file():
    target_file = '../FinAgents/agent_pools/backtest_agent/backtest_agent.py'
    
    with open(target_file, 'r') as f:
        lines = f.readlines()
        
    with open('temp_new_func.py', 'r') as f:
        new_func_lines = f.readlines()
        
    # Find start of the function
    start_idx = -1
    start_marker = '    def run_simple_backtest_paper_interface(self, predictions,'
    
    for i, line in enumerate(lines):
        if line.strip().startswith('def run_simple_backtest_paper_interface'):
            start_idx = i
            break
            
    if start_idx == -1:
        print("Could not find function start")
        return
        
    # Find end (start of __name__ block or end of file)
    end_idx = len(lines)
    
    for i in range(start_idx + 1, len(lines)):
        if line.startswith('if __name__ == "__main__":'):
            end_idx = i
            break
        # Also check if we hit the end of class (dedentation)
        # But hard to detect dedentation if there are empty lines.
        # Look for the next top-level block or class end.
        if line.startswith('class ') or line.startswith('def '): # Top level
             end_idx = i
             break
             
    # Refined search for end: look for `if __name__` specifically as it was at line 2851
    for i in range(start_idx + 1, len(lines)):
        if lines[i].startswith('if __name__ == "__main__":'):
            end_idx = i
            break
            
    print(f"Found start at line {start_idx + 1}")
    print(f"Found end at line {end_idx + 1}")
    
    # Construct new content
    # We keep lines before start
    # Insert new func
    # Keep lines after end
    
    new_content = lines[:start_idx] + new_func_lines + lines[end_idx:]
    
    with open(target_file, 'w') as f:
        f.writelines(new_content)
        
    print("File patched successfully")

if __name__ == '__main__':
    patch_file()
