#!/usr/bin/env python3
"""
FinAgent Memory System - Test Results Summary

This script provides a summary of the latest test results.
"""

import json
import glob
import os
from datetime import datetime

def get_latest_test_results():
    """Get the latest test results file."""
    # Check for latest_test_results.json first
    if os.path.exists("latest_test_results.json"):
        return "latest_test_results.json"
    
    # Otherwise look for test_results_*.json files
    test_files = glob.glob("test_results_*.json")
    if not test_files:
        return None
    
    # Sort by modification time to get the latest
    latest_file = max(test_files, key=os.path.getmtime)
    return latest_file

def format_timestamp(iso_timestamp):
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except:
        return iso_timestamp

def print_summary():
    """Print test results summary."""
    latest_file = get_latest_test_results()
    
    if not latest_file:
        print("❌ No test results found")
        return
    
    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        print("🧪 FINAGENT MEMORY SYSTEM - TEST RESULTS SUMMARY")
        print("=" * 60)
        
        # Basic stats
        print(f"📄 Results File: {latest_file}")
        print(f"📅 Test Time: {format_timestamp(results.get('timestamp', 'Unknown'))}")
        print(f"⏱️  Duration: {results.get('duration', 0):.2f} seconds")
        print()
        
        # Test results
        total = results.get('total_tests', 0)
        passed = results.get('passed_tests', 0)
        failed = results.get('failed_tests', 0)
        success_rate = results.get('success_rate', 0)
        
        print(f"📊 TEST STATISTICS:")
        print(f"   📝 Total Tests: {total}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        print()
        
        # Service connectivity
        connectivity = results.get('connectivity', {})
        print(f"🌐 SERVICE STATUS:")
        for service, status in connectivity.items():
            status_icon = "✅" if status else "❌"
            service_name = {
                'memory': 'Memory Server (8000)',
                'mcp': 'MCP Server (8001)', 
                'a2a': 'A2A Server (8002)'
            }.get(service, service)
            print(f"   {status_icon} {service_name}")
        print()
        
        # Test breakdown by category
        test_results = results.get('test_results', [])
        categories = {}
        
        for test in test_results:
            test_name = test.get('test_name', '')
            passed = test.get('passed', False)
            
            # Categorize tests
            if 'Port' in test_name:
                category = 'Port Connectivity'
            elif 'Database' in test_name:
                category = 'Database Operations'
            elif 'Memory Server' in test_name:
                category = 'Memory Server'
            elif 'MCP' in test_name:
                category = 'MCP Server'
            elif 'A2A' in test_name:
                category = 'A2A Server'
            elif 'Performance' in test_name:
                category = 'Performance'
            else:
                category = 'Other'
            
            if category not in categories:
                categories[category] = {'passed': 0, 'failed': 0, 'tests': []}
            
            if passed:
                categories[category]['passed'] += 1
            else:
                categories[category]['failed'] += 1
            
            categories[category]['tests'].append(test)
        
        print(f"📋 TEST BREAKDOWN BY CATEGORY:")
        for category, stats in categories.items():
            total_cat = stats['passed'] + stats['failed']
            success_rate_cat = (stats['passed'] / total_cat * 100) if total_cat > 0 else 0
            status_icon = "✅" if stats['failed'] == 0 else "⚠️"
            print(f"   {status_icon} {category}: {stats['passed']}/{total_cat} ({success_rate_cat:.0f}%)")
        
        # Failed tests details
        failed_tests = [test for test in test_results if not test.get('passed', True)]
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   • {test.get('test_name', 'Unknown')}")
                if test.get('error'):
                    print(f"     Error: {test['error']}")
        else:
            print(f"\n🎉 ALL TESTS PASSED!")
        
        print()
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error reading test results: {e}")

if __name__ == "__main__":
    print_summary()
