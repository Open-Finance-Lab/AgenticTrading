# Memory System Data Flow Diagrams

## 1. Memory Storage Flow with Auto-Linking

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT STORES NEW MEMORY                            │
└─────────────────────────────────────────────────────────────────────────────┘

Agent calls:
  store_graph_memory(
    query="AAPL bullish momentum detected",
    keywords=["AAPL", "momentum", "bullish"],
    summary="Strong uptrend with volume confirmation",
    agent_id="alpha_pool_01"
  )

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                     STEP 1: VALIDATE & PREPARE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✓ Generate memory_id (UUID)                                              │
│  ✓ Set timestamp (current UTC)                                            │
│  ✓ Create content structure:                                              │
│    {                                                                       │
│      "query": "AAPL bullish...",                                          │
│      "summary": "Strong uptrend...",                                      │
│      "keywords": ["AAPL", "momentum", "bullish"],                         │
│      "event_type": "USER_QUERY",                                          │
│      "session_id": null,                                                  │
│      "correlation_id": null                                               │
│    }                                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                  STEP 2: CREATE MEMORY NODE IN NEO4J                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Cypher Query:                                                             │
│  CREATE (m:Memory {                                                        │
│    memory_id: "uuid-1234-5678",                                           │
│    agent_id: "alpha_pool_01",                                             │
│    memory_type: "user_query",                                             │
│    content: "{...full JSON...}",                                          │
│    content_text: "aapl bullish momentum...",    ◄─ Full-text indexed      │
│    summary: "Strong uptrend...",                ◄─ Full-text indexed      │
│    keywords: ["AAPL", "momentum", "bullish"],  ◄─ Array indexed          │
│    timestamp: datetime("2024-06-15T10:30:00Z"), ◄─ Indexed               │
│    event_type: "USER_QUERY",                                             │
│    log_level: "INFO",                                                     │
│    created_at: datetime(),                                                │
│    lookup_count: 0                                                        │
│  })                                                                        │
│  RETURN m.memory_id as stored_id                                          │
│                                                                             │
│  Result: Memory node successfully created                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│               STEP 3: FIND SIMILAR MEMORIES (AUTO-LINKING)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Method: Intelligent Indexer (Semantic + Keyword)                          │
│                                                                             │
│  Process:                                                                  │
│  1. Extract keywords from query/summary                                    │
│     └─ ["AAPL", "momentum", "bullish"]                                    │
│                                                                             │
│  2. Search for memories with matching keywords                             │
│     Cypher:                                                                │
│     MATCH (m:Memory)                                                       │
│     WHERE ANY(kw IN m.keywords WHERE kw IN $keywords)                     │
│     RETURN m.memory_id, m.summary, m.keywords                             │
│     ORDER BY m.timestamp DESC                                             │
│     LIMIT 10                                                               │
│                                                                             │
│  3. Calculate semantic similarity (if embeddings available)                │
│     └─ Compare text embeddings of summaries                               │
│                                                                             │
│  Similar memories found:                                                   │
│  ├─ memory-2: "AAPL momentum breakout" (100% keyword match)               │
│  ├─ memory-5: "Tech momentum strategy" (80% semantic similarity)          │
│  ├─ memory-8: "AAPL Q2 strength" (75% semantic similarity)                │
│  └─ memory-12: "Momentum trading rules" (65% semantic similarity)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│        STEP 4: CREATE SIMILARITY RELATIONSHIPS (GRAPH LINKING)             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Cypher Query (Batch):                                                     │
│  MATCH (new_m:Memory {memory_id: "uuid-1234-5678"})                       │
│  UNWIND ["memory-2", "memory-5", "memory-8"] AS target_id                 │
│  MATCH (old_m:Memory {memory_id: target_id})                              │
│  MERGE (new_m)-[r:SIMILAR_TO]->(old_m)                                    │
│  RETURN count(r) as links_created                                         │
│                                                                             │
│  Result: 3 SIMILAR_TO relationships created                                │
│                                                                             │
│  Graph now looks like:                                                     │
│                                                                             │
│    ┌─────────────────────┐                                                │
│    │  New Memory         │                                                │
│    │ "AAPL bullish"      │                                                │
│    │ [memory-1234]       │                                                │
│    └──────┬──────────────┘                                                │
│           │                                                               │
│      SIMILAR_TO (3 edges)                                                 │
│      ┌──────┬──────┬──────┐                                               │
│      ▼      ▼      ▼      ▼                                               │
│    M-2    M-5    M-8    M-12                                              │
│    ────────────────────────                                               │
│    "AAPL  "Tech  "AAPL  "Mom                                              │
│    moment momentum Q2    ent                                              │
│    um"    strategy" str"  trad"                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│           STEP 5: INDEX MEMORY (IF INDEXER AVAILABLE)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Intelligent Indexer:                                                      │
│  ├─ Create text embedding (using SentenceTransformer)                     │
│  │  └─ Convert summary to 384-dim vector                                  │
│  ├─ Extract metadata features                                              │
│  │  └─ agent_id, event_type, timestamp features                          │
│  ├─ Calculate performance score (if historical data)                       │
│  │  └─ How often has this been accessed?                                  │
│  └─ Store in memory_index.pkl                                              │
│                                                                             │
│  Index Entry:                                                              │
│  {                                                                         │
│    "memory_id": "uuid-1234-5678",                                         │
│    "text_embedding": [...384 floats...],                                  │
│    "metadata_features": {                                                 │
│      "agent_id": "alpha_pool_01",                                         │
│      "event_type": "USER_QUERY"                                           │
│    },                                                                      │
│    "keywords": ["AAPL", "momentum", "bullish"],                           │
│    "timestamp": datetime(...),                                             │
│    "performance_score": 0.85                                              │
│  }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│         STEP 6: PUBLISH EVENT TO STREAM PROCESSOR (IF AVAILABLE)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Event published:                                                          │
│  {                                                                         │
│    "event_type": "memory_stored",                                         │
│    "memory_id": "uuid-1234-5678",                                         │
│    "agent_id": "alpha_pool_01",                                           │
│    "timestamp": "2024-06-15T10:30:00Z",                                   │
│    "linked_memories": 3                                                   │
│  }                                                                         │
│                                                                             │
│  Stream Processor:                                                         │
│  └─ Triggers reactive subscribers                                         │
│  └─ Updates real-time dashboards                                          │
│  └─ Logs metrics                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                      RETURN RESPONSE TO AGENT                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  {                                                                         │
│    "memory_id": "uuid-1234-5678",                                         │
│    "agent_id": "alpha_pool_01",                                           │
│    "timestamp": "2024-06-15T10:30:00Z",                                   │
│    "content": {...},                                                       │
│    "linked_memories": [                                                    │
│      {"memory_id": "memory-2", "similarity": 1.0},                        │
│      {"memory_id": "memory-5", "similarity": 0.82},                       │
│      {"memory_id": "memory-8", "similarity": 0.78}                        │
│    ],                                                                      │
│    "status": "success"                                                     │
│  }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Memory Retrieval with Graph Expansion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AGENT SEARCHES FOR MEMORIES                           │
└─────────────────────────────────────────────────────────────────────────────┘

Agent calls:
  retrieve_memory_with_expansion(
    search_query="momentum strategy",
    limit=10
  )

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                 PHASE 1: INITIAL FULL-TEXT SEARCH                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Cypher Query:                                                             │
│  MATCH (m:Memory)                                                          │
│  WHERE m.content_text CONTAINS $search_text                               │
│     OR m.summary CONTAINS $search_text                                     │
│     OR ANY(keyword IN m.keywords WHERE keyword CONTAINS $search_text)      │
│  SET m.lookup_count = m.lookup_count + 1                                  │
│  RETURN m.memory_id, m.summary, m.keywords, m.timestamp                   │
│  ORDER BY m.timestamp DESC                                                │
│  LIMIT 5                                                                   │
│                                                                             │
│  Initial Results (5 memories):                                             │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │ Memory-100: "Momentum trading strategy implementation"          │      │
│  │ Memory-101: "Factor momentum in equity selection"               │      │
│  │ Memory-102: "Price momentum vs. earnings momentum"              │      │
│  │ Memory-103: "Momentum reversion trading rules"                 │      │
│  │ Memory-104: "Momentum indicator setup and signals"              │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  Side effect: lookup_count incremented for tracking                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: GRAPH EXPANSION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For each initial result, traverse graph relationships:                     │
│                                                                             │
│  Cypher Query (for each Memory-X):                                         │
│  MATCH (m:Memory {memory_id: "Memory-100"})-[:SIMILAR_TO|:RELATES_TO]-(related:Memory)  │
│  RETURN related.memory_id, related.summary, related.keywords               │
│  LIMIT 3                                                                   │
│                                                                             │
│  Graph Traversal:                                                          │
│                                                                             │
│    Memory-100 (Initial)                                                    │
│    │                                                                       │
│    ├─ SIMILAR_TO ──► Memory-200 (Momentum signals detected)               │
│    ├─ SIMILAR_TO ──► Memory-201 (Trend following strategies)              │
│    └─ RELATES_TO ──► Memory-202 (Risk management in momentum)             │
│                                                                             │
│    Memory-101 (Initial)                                                    │
│    │                                                                       │
│    ├─ SIMILAR_TO ──► Memory-210 (Factor selection methods)                │
│    ├─ SIMILAR_TO ──► Memory-211 (Equities with momentum edge)             │
│    └─ RELATES_TO ──► Memory-212 (Performance attribution)                 │
│                                                                             │
│    ... (similar for Memory-102, 103, 104)                                 │
│                                                                             │
│  Expanded results: 5 initial × 3 expansions = 15 total memories            │
│  (limited to requested limit)                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: DEDUPLICATE & RANK                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Remove duplicates (if any memory appeared in multiple expansions)          │
│  Rank by relevance:                                                        │
│  ├─ Initial results get higher score (direct match)                        │
│  ├─ Expansion results get lower score (secondary)                          │
│  └─ Sort by score, then by timestamp                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                     RETURN EXPANSION RESULTS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  {                                                                         │
│    "initial_results": 5,                                                  │
│    "expanded_results": 10,                                                │
│    "total_results": 15,                                                   │
│    "memories": [                                                          │
│      {                                                                    │
│        "memory_id": "Memory-100",                                         │
│        "summary": "Momentum trading strategy implementation",              │
│        "is_related": False,              ◄─ Initial match                 │
│        "keywords": ["momentum", "strategy", "trading"],                   │
│        "timestamp": "2024-06-10T09:00:00Z"                               │
│      },                                                                    │
│      {                                                                    │
│        "memory_id": "Memory-200",                                         │
│        "summary": "Momentum signals detected in AAPL",                    │
│        "is_related": True,               ◄─ Expansion                     │
│        "related_to": "Memory-100",       ◄─ Parent memory ID             │
│        "relationship_type": "SIMILAR_TO",                                 │
│        "keywords": ["momentum", "AAPL", "signals"],                       │
│        "timestamp": "2024-06-09T14:30:00Z"                               │
│      },                                                                    │
│      ...more results...                                                    │
│    ]                                                                       │
│  }                                                                         │
│                                                                             │
│  Key Benefits:                                                             │
│  ├─ Initial result: direct match to query                                 │
│  ├─ Expanded result (Memory-200): related via SIMILAR_TO                  │
│  │  └─ Agent learns about AAPL-specific momentum                          │
│  ├─ Expanded result (Memory-201): different angle                         │
│  │  └─ Agent learns about trend-following perspective                     │
│  └─ Expanded result (Memory-202): risk considerations                     │
│     └─ Agent learns about risk management                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Manual Relationship Creation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   AGENT LINKS TWO DISCOVERIES                              │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario:
Agent A discovers "AAPL momentum signal" and notes it "CLARIFIES" 
why "Recent market volatility" occurred.

Agent calls:
  create_relationship(
    source_memory_id="memory-signal-1001",
    target_memory_id="memory-volatility-2002",
    relationship_type="CLARIFIES"
  )

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                         VALIDATE INPUTS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✓ Verify source memory exists                                             │
│  ✓ Verify target memory exists                                             │
│  ✓ Sanitize relationship_type to valid graph format                        │
│    └─ "CLARIFIES" → valid                                                 │
│    └─ "clarifies" → convert to "CLARIFIES"                                │
│    └─ "clarifies-it" → invalid, reject                                    │
│                                                                             │
│  Valid relationship types:                                                 │
│  ├─ CLARIFIES, CONTRADICTS, ENHANCES (semantic)                           │
│  ├─ RELATES_TO, SIMILAR_TO (content-based)                                │
│  ├─ TIME_SEQUENCE (temporal)                                              │
│  └─ others (CREATED, TARGETS, IS_TYPE, HAS_PRIORITY, HAS_PERFORMANCE)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                      CREATE RELATIONSHIP EDGE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Cypher Query:                                                             │
│  MATCH (a:Memory {memory_id: "memory-signal-1001"})                        │
│  MATCH (b:Memory {memory_id: "memory-volatility-2002"})                    │
│  MERGE (a)-[r:CLARIFIES]->(b)                                             │
│  RETURN type(r) as rel_type                                                │
│                                                                             │
│  Result: New directed edge created                                         │
│                                                                             │
│  Graph Update:                                                             │
│                                                                             │
│    Before:                          After:                                │
│    ───────────────────────────────────────────────────────────────        │
│    memory-signal-1001              memory-signal-1001                     │
│                                          │                                │
│    memory-volatility-2002          CLARIFIES (directed)                   │
│                                          │                                │
│    (disconnected)                        ▼                                │
│                                    memory-volatility-2002                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                      RETURN SUCCESS RESPONSE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  {                                                                         │
│    "status": "success",                                                   │
│    "relationship_created": "CLARIFIES",                                   │
│    "source_memory_id": "memory-signal-1001",                              │
│    "target_memory_id": "memory-volatility-2002",                          │
│    "direction": "memory-signal-1001 ──► memory-volatility-2002"           │
│  }                                                                         │
│                                                                             │
│  Now when querying about volatility, related signals will appear          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Filtering & Analytics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STRUCTURED MEMORY FILTERING                           │
└─────────────────────────────────────────────────────────────────────────────┘

Agent calls:
  filter_graph_memories(
    filters={
      "agent_id": "alpha_pool_01",
      "event_type": "ERROR",
      "log_level": "WARNING",
      "start_time": "2024-06-01T00:00:00Z",
      "end_time": "2024-06-30T23:59:59Z"
    },
    limit=100,
    offset=0
  )

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                    BUILD DYNAMIC CYPHER QUERY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHERE clauses built from filters:                                         │
│  ├─ m.agent_id = "alpha_pool_01"                                          │
│  ├─ m.event_type = "ERROR"                                                │
│  ├─ m.log_level = "WARNING"                                               │
│  ├─ m.timestamp >= datetime("2024-06-01T00:00:00Z")                        │
│  └─ m.timestamp <= datetime("2024-06-30T23:59:59Z")                        │
│                                                                             │
│  Full Query:                                                               │
│  MATCH (m:Memory)                                                          │
│  WHERE m.agent_id = $agent_id                                             │
│    AND m.event_type = $event_type                                         │
│    AND m.log_level = $log_level                                           │
│    AND m.timestamp >= datetime($start_time)                               │
│    AND m.timestamp <= datetime($end_time)                                 │
│  RETURN m.memory_id, m.summary, m.content, m.timestamp, m.event_type      │
│  ORDER BY m.timestamp DESC                                                │
│  SKIP 0 LIMIT 100                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXECUTE & AGGREGATE RESULTS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Database returns matching records:                                        │
│                                                                             │
│  Total matches: 45 (but limited to 100)                                   │
│  ├─ Offset: 0 (start from first)                                          │
│  ├─ Limit: 100 (return up to 100)                                         │
│  └─ Returned: 45 (actual count)                                           │
│                                                                             │
│  Sample Results:                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │ Time       │ Error Type           │ Summary                      │     │
│  ├──────────────────────────────────────────────────────────────────┤     │
│  │ 2024-06-28 │ MODEL_INFERENCE_ERR  │ Failed to load AAPL model   │     │
│  │ 2024-06-25 │ DATA_FETCH_ERROR     │ Market data unavailable     │     │
│  │ 2024-06-22 │ CALCULATION_ERROR    │ NaN in returns calculation  │     │
│  │ 2024-06-19 │ TIMEOUT_ERROR        │ Database query timeout      │     │
│  │ 2024-06-15 │ VALIDATION_ERROR     │ Signal constraints violated │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                    ANALYTICS: GET STATISTICS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Agent calls:                                                              │
│  get_graph_memory_statistics()                                             │
│                                                                             │
│  Database Queries:                                                         │
│                                                                             │
│  Query 1: Count totals                                                     │
│  ──────────────────────────────────────────────────────────────────        │
│  MATCH (m:Memory)                                                          │
│  RETURN count(m) as memory_count                                           │
│                                                                             │
│  Query 2: Memory type distribution                                         │
│  ──────────────────────────────────────────────────────────────────        │
│  MATCH (m:Memory)                                                          │
│  RETURN m.memory_type as type, count(m) as count                           │
│                                                                             │
│  Query 3: Agent activity ranking                                           │
│  ──────────────────────────────────────────────────────────────────        │
│  MATCH (m:Memory)                                                          │
│  WHERE m.agent_id IS NOT NULL                                             │
│  WITH m.agent_id as agent, count(m) as activity                           │
│  ORDER BY activity DESC LIMIT 10                                           │
│  RETURN agent, activity                                                    │
│                                                                             │
│  Aggregated Response:                                                      │
│  {                                                                         │
│    "total_memories": 5432,                                                │
│    "total_agents": 12,                                                    │
│    "total_relationships": 18932,                                          │
│    "memory_types": {                                                      │
│      "SIGNAL": 2100,                                                      │
│      "ERROR": 450,                                                        │
│      "LEARNING": 1200,                                                    │
│      "PERFORMANCE": 682                                                   │
│    },                                                                      │
│    "agent_activity": {                                                    │
│      "alpha_pool_01": 2100,                                               │
│      "alpha_pool_02": 1980,                                               │
│      "risk_agent": 450,                                                   │
│      ...                                                                   │
│    },                                                                      │
│    "indexer_available": true,                                             │
│    "stream_processor_available": true                                     │
│  }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Why This is Better Than Simple Vector RAG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL VECTOR RAG                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query: "AAPL momentum"                                                    │
│                                                                             │
│  Process:                                                                  │
│  ├─ Convert query to embedding: [0.1, 0.3, -0.2, ..., 0.5]               │
│  ├─ Compute cosine similarity with all document embeddings                │
│  ├─ Sort by similarity score                                              │
│  └─ Return top-5 documents                                                │
│                                                                             │
│  Results:                                                                  │
│  ├─ Memory-A: "AAPL trading momentum" (similarity: 0.92)                  │
│  ├─ Memory-B: "Momentum indicators" (similarity: 0.88)                     │
│  ├─ Memory-C: "AAPL earnings momentum" (similarity: 0.85)                 │
│  ├─ Memory-D: "Velocity factor" (similarity: 0.82)                        │
│  └─ Memory-E: "Acceleration metrics" (similarity: 0.80)                    │
│                                                                             │
│  Problems:                                                                 │
│  ├─ ❌ No temporal sequence: When was Memory-C discovered?                │
│  ├─ ❌ No causality: Why did Memory-A matter for Memory-C?                │
│  ├─ ❌ No context: Memory-D's relationship to others unclear              │
│  ├─ ❌ Lost patterns: Time-series connections not visible                 │
│  ├─ ❌ No agent info: Which agent discovered Memory-B?                    │
│  └─ ❌ Static ranking: No learning from usage patterns                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

         VS

┌─────────────────────────────────────────────────────────────────────────────┐
│                  FINAGENT GRAPH MEMORY SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query: "AAPL momentum"                                                    │
│                                                                             │
│  Process:                                                                  │
│  ├─ Full-text search on keywords/summaries (fast)                         │
│  ├─ Find direct matches                                                    │
│  ├─ Traverse SIMILAR_TO relationships (semantic graph)                    │
│  ├─ Traverse RELATES_TO relationships (contextual)                        │
│  ├─ Include temporal chains (TIME_SEQUENCE)                               │
│  └─ Return enriched results with relationship context                     │
│                                                                             │
│  Results with Rich Context:                                               │
│                                                                             │
│  Memory-A: "AAPL trading momentum" (direct match)                         │
│    ├─ Created by: alpha_pool_01                                           │
│    ├─ When: June 15, 2024                                                 │
│    ├─ Accessed: 47 times (high value)                                     │
│    └─ Connected to:                                                       │
│        ├─ [SIMILAR_TO] Memory-K: "AAPL Q3 strength"                       │
│        │   ├─ Same agent, 1 week later                                    │
│        │   └─ Shows how signal evolved                                     │
│        ├─ [RELATES_TO] Memory-L: "Tech sector rally"                      │
│        │   └─ Broader context                                              │
│        └─ [TIME_SEQUENCE] Memory-M: "AAPL entry signal"                   │
│            └─ Previous signal in sequence                                  │
│                                                                             │
│  Memory-B: "Momentum indicators" (related)                                 │
│    └─ Created by: risk_agent                                              │
│        └─ Shows multi-agent knowledge sharing                              │
│                                                                             │
│  Memory-C: "AAPL earnings momentum" (SIMILAR_TO Memory-A)                 │
│    ├─ Linked because keyword "AAPL" + "momentum"                          │
│    ├─ Created 2 days before Memory-A                                      │
│    ├─ [CLARIFIES] relationship shows it explains Memory-A                 │
│    └─ Historical causality visible                                         │
│                                                                             │
│  Benefits:                                                                 │
│  ├─ ✓ Temporal sequences: Can trace signal evolution                     │
│  ├─ ✓ Causal chains: See why Memory-C led to Memory-A                    │
│  ├─ ✓ Rich context: Multiple relationship types explain connections      │
│  ├─ ✓ Multi-agent: Learn from other agents' discoveries                  │
│  ├─ ✓ Usage patterns: Track which memories are valuable                  │
│  ├─ ✓ Adaptive: System improves with more usage                          │
│  └─ ✓ Explainable: Agent can understand why results were returned       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Multi-Agent Coordination Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT KNOWLEDGE SHARING                            │
└──────────────────────────────────────────────────────────────────────────────┘

Time T=1:
┌────────────────────┐                    Shared Memory Graph
│   Alpha Agent 1    │
│                    │
│ Discovers:        ├──► store_graph_memory({
│ "AAPL bullish"     │      agent_id: "alpha_pool_01",
│ momentum signal    │      query: "AAPL bullish...",
│                    │      keywords: ["AAPL", "momentum"],
│ score: 0.87        │      ...
│                    │    })
└────────────────────┘
                           ┌─────────────────────────┐
                           │  Memory Node: A1        │
                           │  ├─ AAPL bullish signal │
                           │  ├─ By: alpha_pool_01   │
                           │  ├─ Score: 0.87         │
                           │  └─ Time: T=1           │
                           └─────────────────────────┘

Time T=2:
┌────────────────────┐
│   Risk Agent       │                   (Same graph)
│                    │
│ Searching for:     ├──► retrieve_memory_with_expansion({
│ "AAPL signal"      │      search_query: "AAPL",
│                    │      limit: 10
│ (risk check)       │    })
│                    │
└────────────────────┘
                           ┌──────────────────────────────┐
                           │  Results returned:           │
                           │  ├─ Memory A1 (found)        │
                           │  │  ├─ From: alpha_pool_01   │
                           │  │  ├─ Score: 0.87           │
                           │  │  ├─ Accessed: 1 time      │
                           │  │  └─ Signal strength: HIGH  │
                           │  │                            │
                           │  └─ Related discoveries:      │
                           │     └─ (none yet)             │
                           └──────────────────────────────┘

Time T=3:
┌────────────────────┐
│   Alpha Agent 2    │
│                    │
│ Discovers:        ├──► store_graph_memory({
│ "AAPL follow-up"   │      agent_id: "alpha_pool_02",
│ in earnings data   │      query: "AAPL earnings...",
│                    │      keywords: ["AAPL", "earnings"],
│ score: 0.82        │      ...
│                    │    })
└────────────────────┘
                           ┌──────────────────────────┐
                           │  Memory Node: A2         │
                           │  ├─ AAPL earnings signal │
                           │  ├─ By: alpha_pool_02    │
                           │  ├─ Score: 0.82          │
                           │  └─ Time: T=3            │
                           │                          │
                           │  Auto-linked to A1:      │
                           │  └─ [SIMILAR_TO] ──► A1  │
                           │     (both about AAPL)    │
                           └──────────────────────────┘

Time T=4:
┌────────────────────┐
│   Risk Agent       │                   (Updated graph)
│   (continued)      │
│                    ├──► retrieve_memory_with_expansion({
│ Same query:        │      search_query: "AAPL",
│ "AAPL signal"      │      limit: 10
│                    │    })
│ (risk check)       │
│                    │
└────────────────────┘
                           ┌──────────────────────────────┐
                           │  Results NOW INCLUDE:        │
                           │  ├─ Memory A1 (original)     │
                           │  │  ├─ From: alpha_pool_01   │
                           │  │  ├─ Score: 0.87           │
                           │  │  ├─ Accessed: 2 times     │
                           │  │  └─ Related to A2 now!    │
                           │  │                            │
                           │  └─ Memory A2 (new!)         │
                           │     ├─ From: alpha_pool_02    │
                           │     ├─ Score: 0.82            │
                           │     ├─ Similarity: SIMILAR_TO │
                           │     └─ Additional context     │
                           │                              │
                           │  Benefits:                   │
                           │  ✓ Risk agent learns that   │
                           │    A2's finding supports A1 │
                           │  ✓ Combined confidence: 0.87 │
                           │    ✓ both agents agree      │
                           │  ✓ Better decision-making   │
                           └──────────────────────────────┘

Key Insight:
══════════════════════════════════════════════════════════════════════════════
Without explicit communication, Risk Agent benefited from Alpha Agent's 
discoveries. The graph automatically linked similar findings, enabling implicit
multi-agent coordination through the shared knowledge base.

This is why the system is powerful:
├─ Agents don't need to explicitly tell each other about discoveries
├─ Graph relationships enable automatic knowledge aggregation
├─ System improves as more agents contribute
└─ Collective intelligence emerges from individual contributions
```

