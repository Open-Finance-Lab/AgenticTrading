#!/usr/bin/env python3
"""
Neo4j Connection Diagnostic Script
"""

import sys
import time

try:
    from neo4j import GraphDatabase
    print("✅ Neo4j driver is available")
except ImportError:
    print("❌ Neo4j driver not available. Install with: pip install neo4j")
    sys.exit(1)

# Test configurations
test_configs = [
    {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "finagent123"},
    {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "FinOrchestration"},
    {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "neo4j"},
    {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "password"},
    {"uri": "bolt://127.0.0.1:7687", "username": "neo4j", "password": "finagent123"},
]

print("🔍 Testing Neo4j connections...")
print(f"Timestamp: {time.ctime()}")

for i, config in enumerate(test_configs, 1):
    print(f"\n🧪 Test {i}: {config['uri']} with user '{config['username']}' and password '{config['password']}'")
    
    try:
        driver = GraphDatabase.driver(
            config["uri"],
            auth=(config["username"], config["password"])
        )
        
        # Test the connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test, datetime() as timestamp")
            record = result.single()
            test_value = record["test"]
            timestamp = record["timestamp"]
            
            if test_value == 1:
                print(f"✅ SUCCESS: Connected successfully!")
                print(f"   Server timestamp: {timestamp}")
                
                # Try to get server info
                try:
                    info_result = session.run("CALL dbms.components() YIELD name, versions")
                    components = list(info_result)
                    if components:
                        print(f"   Server info: {components[0]['name']} {components[0]['versions'][0]}")
                except:
                    print("   Server info not available")
                
                driver.close()
                print(f"✅ Working configuration found: {config}")
                sys.exit(0)
            else:
                print(f"❌ FAILED: Unexpected test result: {test_value}")
                
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        
    finally:
        try:
            driver.close()
        except:
            pass
    
    # Small delay to avoid rate limiting
    time.sleep(0.5)

print("\n❌ No working configuration found")
print("🔧 Troubleshooting suggestions:")
print("   1. Check if Neo4j is running: ps aux | grep neo4j")
print("   2. Check if port 7687 is listening: netstat -an | grep 7687")
print("   3. Try connecting via Neo4j Browser at http://localhost:7474")
print("   4. Check Neo4j logs for authentication errors")
print("   5. Reset password using Neo4j Desktop or cypher-shell")
