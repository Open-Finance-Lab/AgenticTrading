# Memory System - Quick Reference Guide

## System Quick Start

### Three Entry Points for Agents

```
┌─────────────────────────────────────────────────────────────┐
│                   How Agents Access Memory                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Option 1: MCP Protocol (Recommended for LLMs)             │
│  ═════════════════════════════════════════════════════════ │
│  ├─ Use: Any OpenAI GPT or compatible LLM                  │
│  ├─ Port: 8001                                             │
│  ├─ Protocol: JSON-RPC 2.0                                 │
│  │                                                         │
│  │  Example tool call (from LLM):                          │
│  │  {                                                      │
│  │    "name": "retrieve_graph_memory",                     │
│  │    "arguments": {                                       │
│  │      "search_query": "AAPL momentum",                   │
│  │      "limit": 5                                         │
│  │    }                                                    │
│  │  }                                                      │
│  │                                                         │
│  └─ Best for: LLM agents, tool calling, auto-orchestration│
│                                                             │
│  Option 2: HTTP REST API                                   │
│  ════════════════════════════════════════════════════════ │
│  ├─ Use: Python agents, custom scripts, web apps           │
│  ├─ Port: 8000                                             │
│  ├─ Protocol: HTTP POST/GET                                │
│  │                                                         │
│  │  Example call:                                          │
│  │  POST /memory/store                                     │
│  │  Content-Type: application/json                         │
│  │  {                                                      │
│  │    "query": "AAPL bullish",                             │
│  │    "keywords": ["AAPL", "bullish"],                     │
│  │    "summary": "Strong uptrend",                         │
│  │    "agent_id": "alpha_pool_01"                          │
│  │  }                                                      │
│  │                                                         │
│  └─ Best for: Direct integrations, custom protocols        │
│                                                             │
│  Option 3: A2A Protocol (Inter-Agent Communication)        │
│  ════════════════════════════════════════════════════════ │
│  ├─ Use: Agent pools, distributed systems                  │
│  ├─ Port: 8002                                             │
│  ├─ Protocol: Agent-to-Agent standard                      │
│  │                                                         │
│  │  Example request:                                       │
│  │  {                                                      │
│  │    "to": "memory_agent",                                │
│  │    "action": "search",                                  │
│  │    "query": "momentum strategy",                        │
│  │    "limit": 10                                          │
│  │  }                                                      │
│  │                                                         │
│  └─ Best for: Multi-agent systems, agent pool coordination │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Tool Reference Card

### Storing Memories

```python
# STORE_GRAPH_MEMORY - Save a discovery to graph
store_graph_memory(
    query: str,                    # What was discovered
    keywords: list[str],           # ["AAPL", "momentum", ...]
    summary: str,                  # 1-2 sentence summary
    agent_id: str,                 # Which agent found this
    event_type: str = "USER_QUERY",  # Type: SIGNAL, ERROR, LEARNING, etc.
    log_level: str = "INFO",         # Severity: INFO, WARNING, ERROR
    session_id: str = None,          # Group related events
    correlation_id: str = None       # Link cause-effect chains
)

# Returns:
# {
#   "memory_id": "uuid-xxx",
#   "linked_memories": 3,  # How many similar ones auto-linked
#   "status": "success"
# }

# Example:
store_graph_memory(
    query="AAPL momentum signal detected with 0.87 confidence",
    keywords=["AAPL", "momentum", "bullish", "0.87"],
    summary="Strong momentum in AAPL with volume confirmation",
    agent_id="alpha_pool_01",
    event_type="SIGNAL",
    log_level="INFO"
)
```

### Retrieving Memories

```python
# RETRIEVE_GRAPH_MEMORY - Fast text search (no relationships)
retrieve_graph_memory(
    search_query: str,      # "momentum AAPL"
    limit: int = 5          # How many results
)

# Use when: You want quick, direct matches
# Search on: keywords, summary, full content text
# Speed: Very fast (~50ms for 100K memories)
# Context: Direct matches only

# Example:
retrieve_graph_memory(
    search_query="AAPL momentum strategy",
    limit=5
)

─────────────────────────────────────────────────────────────

# RETRIEVE_MEMORY_WITH_EXPANSION - Search + relationship traversal
retrieve_memory_with_expansion(
    search_query: str,      # "momentum strategy"
    limit: int = 10         # Total combined results
)

# Use when: You want context and related findings
# Returns: Direct matches + connected memories via:
#   ├─ SIMILAR_TO (semantic neighbors)
#   ├─ RELATES_TO (contextual connections)
#   └─ TIME_SEQUENCE (temporal chains)
# Speed: Moderate (~200-300ms)
# Context: Rich with relationship info

# Example:
retrieve_memory_with_expansion(
    search_query="how does momentum trading fail?",
    limit=20  # 5 direct + 15 expansion
)
```

### Finding Specific Memory Types

```python
# FILTER_GRAPH_MEMORIES - Structured queries
filter_graph_memories(
    filters: {
        "agent_id": "alpha_pool_01",           # Which agent
        "event_type": "ERROR",                 # What happened
        "log_level": "WARNING",                # How serious
        "start_time": "2024-06-01T00:00:00Z", # Date range
        "end_time": "2024-06-30T23:59:59Z",
        "session_id": "session-abc123"         # Grouped events
    },
    limit: int = 100,
    offset: int = 0
)

# Use when: You need structured/analytical queries
# Perfect for: Timeline analysis, error tracking, auditing

# Example - Find all errors from a specific agent:
filter_graph_memories(
    filters={
        "agent_id": "alpha_pool_01",
        "event_type": "ERROR",
        "log_level": "WARNING"
    },
    limit=50
)

# Example - Find everything from last week:
filter_graph_memories(
    filters={
        "start_time": "2024-06-08T00:00:00Z",
        "end_time": "2024-06-14T23:59:59Z"
    },
    limit=100
)
```

### Creating Links Between Memories

```python
# CREATE_RELATIONSHIP - Manual semantic linking
create_relationship(
    source_memory_id: str,      # Starting memory (UUID)
    target_memory_id: str,      # Ending memory (UUID)
    relationship_type: str      # Type of connection
)

# Relationship types available:
# ├─ CLARIFIES: "This explains that"
# ├─ CONTRADICTS: "This disputes that"
# ├─ ENHANCES: "This improves that"
# ├─ RELATES_TO: "This is related to that"
# ├─ SIMILAR_TO: "This is similar to that"
# ├─ TIME_SEQUENCE: "This happened before that"
# ├─ CREATED: "Agent created this"
# ├─ TARGETS: "This targets that"
# └─ IS_TYPE: "This is a type of that"

# Returns:
# {
#   "status": "success",
#   "relationship_created": "CLARIFIES"
# }

# Example:
create_relationship(
    source_memory_id="memory-signal-1001",
    target_memory_id="memory-volatility-2002",
    relationship_type="CLARIFIES"
)
```

### System Statistics

```python
# GET_GRAPH_MEMORY_STATISTICS - System health check
get_graph_memory_statistics()

# Returns:
# {
#   "total_memories": 5432,
#   "total_agents": 12,
#   "total_relationships": 18932,
#   "memory_types": {
#       "SIGNAL": 2100,
#       "ERROR": 450,
#       "LEARNING": 1200,
#       "PERFORMANCE": 682
#   },
#   "agent_activity": {
#       "alpha_pool_01": 2100,
#       "alpha_pool_02": 1980,
#       "risk_agent": 450
#   },
#   "indexer_available": true,
#   "stream_processor_available": true
# }

# Use for: Monitoring, capacity planning, health checks
```

### Maintenance

```python
# PRUNE_GRAPH_MEMORIES - Clean old/irrelevant memories
prune_graph_memories(
    max_age_days: int = 180,       # Delete if older than this
    min_lookup_count: int = 1      # Keep if accessed at least once
)

# Returns:
# {
#   "deleted_count": 342,
#   "retained_count": 5090,
#   "status": "success"
# }

# Schedule: Run weekly or monthly
# Policy: Keep accessed memories indefinitely, delete old unused
```

---

## Decision Trees: Which Tool to Use?

### When to Retrieve Memories

```
┌─ Do you need fast results?
│  └─ YES: Use retrieve_graph_memory()
│          └─ Direct text search
│          └─ ~50ms latency
│
└─ Do you need context and relationships?
   └─ YES: Use retrieve_memory_with_expansion()
           └─ Includes similar/related findings
           └─ ~200-300ms latency

┌─ Do you need filtered/analytical data?
│  └─ YES: Use filter_graph_memories()
│          └─ Agent history, error timeline, etc.
│          └─ ~100-200ms latency
│
└─ Do you need semantic embedding search?
   └─ YES: Use semantic_search_memories()
           └─ Deep similarity matching
           └─ ~500-1000ms latency
```

### Memory Storage Decision

```
Every time an agent completes significant work:
├─ Discovers a signal/pattern
│  └─ Call: store_graph_memory(
│     event_type="SIGNAL"
│  )
│
├─ Encounters an error
│  └─ Call: store_graph_memory(
│     event_type="ERROR",
│     log_level="WARNING"
│  )
│
├─ Learns something valuable
│  └─ Call: store_graph_memory(
│     event_type="LEARNING"
│  )
│
└─ Completes a trade/action
   └─ Call: store_graph_memory(
      event_type="PERFORMANCE"
   )

The system will automatically link similar findings.
```

### Linking Decisions

```
When you notice a relationship between two memories:
├─ One discovery explains another
│  └─ Use: CLARIFIES
│
├─ One contradicts another
│  └─ Use: CONTRADICTS
│
├─ One improves another
│  └─ Use: ENHANCES
│
├─ One is similar to another
│  └─ Use: SIMILAR_TO (usually auto-created)
│
└─ One happened before another
   └─ Use: TIME_SEQUENCE

Manual linking is optional - system does it automatically.
Use for edge cases or important semantic links.
```

---

## Performance Expectations

### Query Latency

| Operation | Database Size | Latency | Notes |
|-----------|---------------|---------|-------|
| Full-text search | < 100K | ~50ms | Fastest |
| Keyword search | < 100K | ~75ms | Very fast |
| Graph expansion | < 100K | ~200ms | Multiple traversals |
| Semantic search | < 100K | ~800ms | Embedding computation |
| Filter queries | < 100K | ~100ms | Depends on filter complexity |
| | | | |
| Full-text search | 100K-1M | ~150ms | Indexes help |
| Graph expansion | 100K-1M | ~500ms | More nodes to traverse |
| Semantic search | 100K-1M | ~2s | Scale issue |

### Storage Performance

| Operation | Records | Latency | Notes |
|-----------|---------|---------|-------|
| Single store | 1 | ~50ms | Includes auto-linking |
| Batch store | 100 | ~200ms | ~2ms per record |
| Batch store | 1000 | ~1.5s | Scales linearly |

### Scale Limits (Single Neo4j Instance)

| Metric | Recommended Max | Impact |
|--------|-----------------|--------|
| Total memories | 1M | Graph traversal becomes slow |
| Relationships/memory | 10 | Storage grows cubically |
| Agent count | 100 | Index size manageable |
| Concurrent users | 50 | Connection pool saturates |

---

## Common Patterns

### Pattern 1: Agent Learning from History

```python
async def learn_from_past():
    # Step 1: Search for past discoveries
    past = await retrieve_memory_with_expansion(
        search_query="momentum trading failure",
        limit=10
    )
    
    # Step 2: Analyze patterns
    patterns = analyze_patterns(past)
    
    # Step 3: Store learning
    await store_graph_memory(
        query=f"Learned: {patterns.summary}",
        keywords=patterns.keywords,
        summary=patterns.description,
        agent_id="my_agent",
        event_type="LEARNING"
    )
    
    return patterns
```

### Pattern 2: Error Tracking & Root Cause

```python
async def handle_error(error):
    # Step 1: Store the error
    await store_graph_memory(
        query=f"Error: {error.message}",
        keywords=["error", error.type],
        summary=error.message,
        agent_id="my_agent",
        event_type="ERROR",
        log_level="WARNING"
    )
    
    # Step 2: Find similar errors
    similar = await retrieve_memory_with_expansion(
        search_query=error.type,
        limit=5
    )
    
    # Step 3: Find root cause from history
    root_cause = find_root_cause(similar)
    
    # Step 4: Link error to root cause
    if root_cause:
        await create_relationship(
            source_memory_id=current_error_id,
            target_memory_id=root_cause.memory_id,
            relationship_type="RELATES_TO"
        )
    
    return root_cause
```

### Pattern 3: Multi-Agent Consensus

```python
async def get_consensus_signal():
    # Step 1: Search for related signals
    signals = await retrieve_memory_with_expansion(
        search_query="AAPL momentum signal",
        limit=20
    )
    
    # Step 2: Group by agent
    by_agent = group_by_agent(signals)
    
    # Step 3: Calculate confidence
    confidence = calculate_consensus(by_agent)
    
    # Step 4: Store final decision
    if confidence > 0.8:
        await store_graph_memory(
            query="AAPL consensus signal: BUY",
            keywords=["AAPL", "consensus", "BUY"],
            summary=f"Multi-agent consensus at {confidence}",
            agent_id="consensus_agent",
            event_type="SIGNAL"
        )
    
    return confidence
```

### Pattern 4: Temporal Analysis

```python
async def analyze_temporal_pattern():
    # Step 1: Get signal from 3 months ago
    past_signal = await retrieve_graph_memory(
        search_query="AAPL momentum",
        limit=1
    )
    
    # Step 2: Get follow-up discovery
    follow_up = await retrieve_memory_with_expansion(
        search_query=f"AAPL after {past_signal.timestamp}",
        limit=5
    )
    
    # Step 3: Link temporal sequence
    for mem in follow_up:
        await create_relationship(
            source_memory_id=past_signal.memory_id,
            target_memory_id=mem.memory_id,
            relationship_type="TIME_SEQUENCE"
        )
    
    # Step 4: Store pattern discovery
    await store_graph_memory(
        query="AAPL momentum persists across quarters",
        keywords=["AAPL", "temporal", "persistence"],
        summary="Signal impact lasts 3+ months",
        agent_id="analysis_agent",
        event_type="LEARNING"
    )
```

---

## Troubleshooting

### Issue: Slow retrieval queries

**Cause:** Large database with many relationships
**Solution:** 
- Use `retrieve_graph_memory()` instead of `retrieve_memory_with_expansion()`
- Reduce the limit parameter
- Use `filter_graph_memories()` for specific agent/event_type

### Issue: Duplicate linking

**Cause:** Same memory linked multiple times via auto-linking
**Solution:**
- System uses MERGE in Cypher, which prevents duplicates
- Check if relationships are actually duplicates or different types

### Issue: Memory not found

**Cause:** Keywords don't match, memory deleted, search too specific
**Solution:**
- Try broader search terms
- Use `filter_graph_memories()` with agent_id/time range
- Check if memory exists with `get_graph_memory_statistics()`

### Issue: Out of memory / Slow Neo4j

**Cause:** Database reached scale limits
**Solution:**
- Run `prune_graph_memories()` to clean old data
- Disable semantic indexer if not needed
- Consider Neo4j clustering/sharding

---

## Security & Access Control

```
Current implementation: No authentication
├─ All tools callable by all agents
├─ Agent_id is user-supplied (not verified)
└─ Good for trusted environments (internal networks)

For production:
├─ Add OAuth2/JWT validation on servers
├─ Implement role-based access control (RBAC)
├─ Audit all memory access
├─ Encrypt sensitive content at rest
└─ Use separate Neo4j credentials with minimal privileges
```

---

## Integration Examples

### LLM Agent with Tools

```python
from mcp.client.session import ClientSession

async def gpt_agent_with_memory():
    # Initialize MCP session
    mcp_session = await ClientSession().create()
    
    # Your memory tools are available
    tools = [
        "store_graph_memory",
        "retrieve_graph_memory",
        "retrieve_memory_with_expansion",
        "create_relationship",
        # ... others
    ]
    
    # Make API call with tools
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Find AAPL momentum strategies"}
        ],
        tools=tools,
        tool_choice="auto"
    )
    
    # LLM will call tools automatically
    # Results flow back into context
```

### Python Agent Script

```python
import asyncio
from unified_database_manager import UnifiedDatabaseManager

async def python_agent():
    # Initialize database manager
    db = UnifiedDatabaseManager(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="FinOrchestration"
    )
    
    await db.connect()
    
    try:
        # Store discovery
        result = await db.store_memory(
            query="AAPL signal",
            keywords=["AAPL", "signal"],
            summary="Strong signal detected",
            agent_id="my_agent"
        )
        
        # Retrieve results
        memories = await db.retrieve_memory("momentum", limit=5)
        
        # Get expanded context
        expanded = await db.retrieve_memory_with_expansion(
            "momentum", limit=10
        )
        
    finally:
        await db.close()

asyncio.run(python_agent())
```

---

## Summary

The FinAgent Memory System provides:

✅ **Graph-based knowledge storage** - Not just vectors, but relationships
✅ **Multi-protocol access** - MCP, HTTP, A2A for different agents
✅ **Automatic linking** - Similar memories connected without explicit instruction
✅ **Rich querying** - Full-text, semantic, filtered, and graph traversal
✅ **Multi-agent learning** - Agents benefit from each other's discoveries
✅ **Temporal tracking** - Understand how discoveries evolve over time
✅ **Error tracking** - Learn from failures and root causes
✅ **High performance** - 50-200ms latency for most queries on typical workloads

Use it to build **intelligent, self-improving multi-agent systems**.

