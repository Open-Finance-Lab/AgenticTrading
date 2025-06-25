#!/usr/bin/env python3
"""
Verification script to ensure all Chinese characters have been removed
from the AutonomousAgent codebase and related files.
"""

import os
import re
import sys

def check_chinese_characters(file_path):
    """Check if a file contains Chinese characters"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match Chinese characters
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        matches = chinese_pattern.findall(content)
        
        return matches
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def main():
    """Main verification function"""
    print("=== Chinese Character Verification ===")
    print("Checking key files for remaining Chinese characters...")
    
    # Files to check
    files_to_check = [
        '/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/agents/autonomous/autonomous_agent.py',
        '/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/examples/autonomous_agent_example.py',
        '/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/agents/autonomous/README.md'
    ]
    
    total_chinese_found = 0
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            chinese_chars = check_chinese_characters(file_path)
            
            if chinese_chars:
                print(f"\n❌ FOUND Chinese characters in {os.path.basename(file_path)}:")
                for char_group in chinese_chars[:5]:  # Show first 5 occurrences
                    print(f"   - {char_group}")
                total_chinese_found += len(chinese_chars)
            else:
                print(f"✅ {os.path.basename(file_path)} - No Chinese characters found")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print(f"\n=== Verification Summary ===")
    if total_chinese_found == 0:
        print("🎉 SUCCESS: All files are Chinese-free!")
        print("✅ Complete English conversion verified")
        return True
    else:
        print(f"❌ FOUND {total_chinese_found} Chinese character groups")
        print("❌ Manual review and cleanup required")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
