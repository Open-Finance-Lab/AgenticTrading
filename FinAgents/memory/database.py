from neo4j import GraphDatabase, AsyncGraphDatabase
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "FinOrchestration")

logging.basicConfig(level=logging.INFO)

class TradingGraphMemory:
    def __init__(self, uri, user, password):
        try:
            self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
            logging.info("Successfully connected to Neo4j async driver.")
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    async def close(self):
        if self.driver:
            await self.driver.close()
            logging.info("Neo4j connection closed.")

    async def clear_database(self):
        logging.warning("Clearing all data from the database...")
        query = "MATCH (n) DETACH DELETE n"
        async with self.driver.session() as session:
            await session.run(query)
        logging.info("Database has been cleared.")

    async def create_memory_index(self):
        index_name = "memory_search_index"
        query = f"CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS FOR (m:Memory) ON EACH [m.summary, m.keywords]"
        async with self.driver.session() as session:
            await session.run(query)
        logging.info(f"Full-text index '{index_name}' is ready.")

    async def create_structured_indexes(self):
        logging.info("Ensuring structured property indexes are created...")
        async with self.driver.session() as session:
            await session.run("CREATE INDEX memory_event_type IF NOT EXISTS FOR (m:Memory) ON (m.event_type)")
            await session.run("CREATE INDEX memory_log_level IF NOT EXISTS FOR (m:Memory) ON (m.log_level)")
            await session.run("CREATE INDEX memory_session_id IF NOT EXISTS FOR (m:Memory) ON (m.session_id)")
            await session.run("CREATE INDEX memory_agent_id IF NOT EXISTS FOR (m:Memory) ON (m.agent_id)")
        logging.info("Structured property indexes are ready.")

    async def store_memory(
        self, query: str, keywords: List[str], summary: str, agent_id: str,
        event_type: Optional[str] = 'USER_QUERY', log_level: Optional[str] = 'INFO',
        session_id: Optional[str] = None, correlation_id: Optional[str] = None,
        similarity_threshold: float = 1.25
    ):
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        async with self.driver.session() as session:
            # CORRECTED LINE: Added 'await' before session.begin_transaction()
            async with await session.begin_transaction() as tx:
                create_cypher = """
                CREATE (m:Memory {
                    query: $query, keywords: $keywords, summary: $summary, agent_id: $agent_id,
                    memory_id: $memory_id, timestamp: $timestamp, lookup_count: 0,
                    event_type: $event_type, log_level: $log_level, session_id: $session_id,
                    correlation_id: $correlation_id
                }) RETURN m
                """
                parameters = {
                    "query": query, "keywords": keywords, "summary": summary, "agent_id": agent_id,
                    "memory_id": memory_id, "timestamp": timestamp, "event_type": event_type,
                    "log_level": log_level, "session_id": session_id, "correlation_id": correlation_id
                }
                result = await tx.run(create_cypher, parameters)
                create_result = await result.single()
                node = create_result.data()['m'] if create_result else None

                if not node:
                    logging.error("Failed to create the memory node.")
                    await tx.rollback()
                    return None
                
                logging.info(f"Stored memory with ID: {memory_id}")

                search_phrase = summary + " " + " ".join(keywords)
                find_similar_cypher = """
                CALL db.index.fulltext.queryNodes("memory_search_index", $search_phrase) YIELD node, score
                WHERE score >= $threshold AND node.memory_id <> $new_memory_id
                RETURN node.memory_id AS similar_memory_id LIMIT 5
                """
                find_params = {"search_phrase": search_phrase, "threshold": similarity_threshold, "new_memory_id": memory_id}
                similar_records_result = await tx.run(find_similar_cypher, find_params)
                similar_memory_ids = [record["similar_memory_id"] async for record in similar_records_result]
                
                if similar_memory_ids:
                    logging.info(f"Found {len(similar_memory_ids)} similar memories. Creating relationships.")
                    link_cypher = """
                    MATCH (new_m:Memory {memory_id: $new_memory_id})
                    UNWIND $similar_ids AS target_id
                    MATCH (old_m:Memory {memory_id: target_id})
                    MERGE (new_m)-[r:SIMILAR_TO]->(old_m)
                    """
                    link_params = {"new_memory_id": memory_id, "similar_ids": similar_memory_ids}
                    await tx.run(link_cypher, link_params)
            # The transaction is automatically committed on exiting the 'async with' block successfully.

            node_data = dict(node)
            return {
                'query': node_data.get('query'), 'keywords': node_data.get('keywords'), 'summary': node_data.get('summary'),
                'metadata': {k: node_data.get(k) for k in ['agent_id', 'memory_id', 'timestamp', 'lookup_count', 'event_type', 'log_level', 'session_id', 'correlation_id']},
                'linked_memories': similar_memory_ids
            }
        return None



    async def store_memories_batch(self, events: List[Dict[str, Any]]):
        if not events:
            return 0

        for event in events:
            event['memory_id'] = str(uuid.uuid4())
            event['timestamp'] = datetime.now().isoformat()
            event['lookup_count'] = 0

        cypher_query = """
        UNWIND $events AS event
        CREATE (m:Memory)
        SET m = event
        RETURN count(m) as created_count
        """
        async with self.driver.session() as session:
            result = await session.run(cypher_query, parameters={"events": events})
            record = await result.single()
            return record["created_count"] if record else 0

    async def filter_memories(self, filters: Dict[str, Any], limit: int = 100, offset: int = 0):
        where_clauses = []
        params = {'limit': limit, 'offset': offset}

        if filters.get('start_time'):
            where_clauses.append("datetime(m.timestamp) >= datetime($start_time)")
            params['start_time'] = filters['start_time']
        if filters.get('end_time'):
            where_clauses.append("datetime(m.timestamp) <= datetime($end_time)")
            params['end_time'] = filters['end_time']
        if filters.get('event_types'):
            where_clauses.append("m.event_type IN $event_types")
            params['event_types'] = filters['event_types']
        if filters.get('log_levels'):
            where_clauses.append("m.log_level IN $log_levels")
            params['log_levels'] = filters['log_levels']
        if filters.get('session_id'):
            where_clauses.append("m.session_id = $session_id")
            params['session_id'] = filters['session_id']
        if filters.get('agent_id'):
            where_clauses.append("m.agent_id = $agent_id")
            params['agent_id'] = filters['agent_id']

        where_statement = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            MATCH (m:Memory)
            {where_statement}
            RETURN m
            ORDER BY m.timestamp DESC
            SKIP $offset
            LIMIT $limit
        """

        async with self.driver.session() as session:
            result = await session.run(query, params)
            return [dict(record['m']) async for record in result]

    async def get_statistics(self):
        stats = {}
        async with self.driver.session() as session:
            total_res = await session.run("MATCH (m:Memory) RETURN count(m) as count")
            stats['total_memories'] = (await total_res.single())['count']

            by_type_res = await session.run("""
                MATCH (m:Memory) WHERE m.event_type IS NOT NULL
                RETURN m.event_type as type, count(m) as count
            """)
            stats['memories_by_type'] = {r['type']: r['count'] async for r in by_type_res}

            by_level_res = await session.run("""
                MATCH (m:Memory) WHERE m.log_level IS NOT NULL
                RETURN m.log_level as level, count(m) as count
            """)
            stats['memories_by_log_level'] = {r['level']: r['count'] async for r in by_level_res}
        return stats

    async def retrieve_memory(self, search_query: str, limit: int = 5):
        cypher_query = """
        CALL db.index.fulltext.queryNodes("memory_search_index", $search_query) YIELD node, score
        RETURN node, score ORDER BY score DESC LIMIT $limit
        """
        params = {"search_query": search_query, "limit": limit}
        processed_results = []
        
        async with self.driver.session() as session:
            result = await session.run(cypher_query, params)
            result_records = [record async for record in result]

            if not result_records:
                return []

            retrieved_ids = []
            for record in result_records:
                node = record.data()['node']
                score = record.data()['score']
                retrieved_ids.append(node['memory_id'])
                processed_results.append({
                    'score': score,
                    'memory': {
                        'query': node['query'], 'keywords': node['keywords'], 'summary': node['summary'],
                        'metadata': { 'agent_id': node['agent_id'], 'memory_id': node['memory_id'], 'timestamp': node['timestamp'], 'lookup_count': node.get('lookup_count', 0) + 1 },
                    },
                })
            
            increment_query = "UNWIND $memory_ids AS mem_id MATCH (m:Memory {memory_id: mem_id}) SET m.lookup_count = coalesce(m.lookup_count, 0) + 1"
            write_result = await session.run(increment_query, {"memory_ids": retrieved_ids})
            await write_result.consume()

        logging.info(f"Retrieved {len(processed_results)} memories for query: '{search_query}'")
        return processed_results


    async def retrieve_memory_with_expansion(self, search_query: str, limit: int = 10):
        final_results = {}
        
        async with self.driver.session() as session:
            seed_query = """
            CALL db.index.fulltext.queryNodes("memory_search_index", $search_query) YIELD node, score
            RETURN node, score ORDER BY score DESC LIMIT $limit
            """
            seed_params = {"search_query": search_query, "limit": limit}
            seed_result = await session.run(seed_query, seed_params)
            seed_records = [record async for record in seed_result]
            logging.info(f"[Expansion Search] Step 1: Found {len(seed_records)} seed memories.")

            if not seed_records:
                return []
            
            seed_ids = []
            for record in seed_records:
                node = record.data()['node']
                score = record.data()['score']
                memory_id = node['memory_id']
                seed_ids.append(memory_id)
                final_results[memory_id] = {
                    'score': score, 'memory': { 'query': node['query'], 'keywords': node['keywords'], 'summary': node['summary'],
                        'metadata': { 'agent_id': node['agent_id'], 'memory_id': memory_id, 'timestamp': node['timestamp'], 'lookup_count': node.get('lookup_count', 0) },
                    },
                }

            expansion_query = """
            UNWIND $seed_ids AS seedId
            MATCH (seedNode:Memory {memory_id: seedId})-[:SIMILAR_TO]-(relatedNode:Memory)
            WHERE NOT relatedNode.memory_id IN $seed_ids
            RETURN DISTINCT relatedNode
            """
            expansion_params = {"seed_ids": seed_ids}
            expansion_result = await session.run(expansion_query, expansion_params)
            expanded_records = [record async for record in expansion_result]
            logging.info(f"[Expansion Search] Step 2: Found {len(expanded_records)} new memories via expansion.")
            
            min_seed_score = min(rec['score'] for rec in seed_records) if seed_records else 0
            for record in expanded_records:
                node = record.data()['relatedNode']
                memory_id = node['memory_id']
                if memory_id not in final_results:
                    final_results[memory_id] = {
                        'score': min_seed_score - 0.1, 'memory': { 'query': node['query'], 'keywords': node['keywords'], 'summary': node['summary'],
                            'metadata': { 'agent_id': node['agent_id'], 'memory_id': memory_id, 'timestamp': node['timestamp'], 'lookup_count': node.get('lookup_count', 0) },
                        },
                    }
            
            all_ids = list(final_results.keys())
            increment_query = "UNWIND $memory_ids AS mem_id MATCH (m:Memory {memory_id: mem_id}) SET m.lookup_count = coalesce(m.lookup_count, 0) + 1"
            write_result = await session.run(increment_query, {"memory_ids": all_ids})
            await write_result.consume()
            logging.info(f"Incremented lookup count for {len(all_ids)} memories (seed + expansion).")

        sorted_results = sorted(final_results.values(), key=lambda x: x['score'], reverse=True)
        logging.info(f"[Expansion Search] Step 4: Returning {min(limit, len(sorted_results))} combined and ranked results.")
        return sorted_results[:limit]
    
    async def prune_memories(self, max_age_days: int = 180, min_lookup_count: int = 1):
        query = """
        // 1. Find all "anchor" memories that are important enough to keep on their own
        MATCH (anchor:Memory)
        WHERE anchor.lookup_count >= $min_lookup_count
        WITH collect(anchor) AS anchors

        // 2. Find all memories that are directly connected to these anchors
        MATCH (protected_neighbor:Memory)-[:SIMILAR_TO]-(a:Memory)
        WHERE a IN anchors
        WITH anchors + collect(DISTINCT protected_neighbor) as protected_nodes_list
        
        // Unwind the list to get unique nodes
        UNWIND protected_nodes_list as protected_node
        WITH COLLECT(DISTINCT protected_node) as protected_nodes

        // 3. Find all memories that are NOT protected
        MATCH (m:Memory)
        WHERE NOT m IN protected_nodes
        
        // 4. From this unprotected group, find the ones that are old and irrelevant
        WITH m, datetime() - duration({days: $max_age_days}) AS cutoff_datetime
        WHERE datetime(m.timestamp) < cutoff_datetime AND m.lookup_count < $min_lookup_count
        
        // 5. Delete them
        DETACH DELETE m
        RETURN count(m) as deleted_count
        """
        params = {"max_age_days": max_age_days, "min_lookup_count": min_lookup_count}
        
        async with self.driver.session() as session:
            result = await session.run(query, params)
            record = await result.single()
            deleted_count = record['deleted_count'] if record else 0
            logging.info(f"Pruned {deleted_count} old/irrelevant memories, protecting important clusters.")
            return deleted_count

    async def create_relationship(self, source_memory_id: str, target_memory_id: str, relationship_type: str):
        safe_relationship_type = "".join(c for c in relationship_type.upper() if c.isalnum() or c == '_')
        if not safe_relationship_type:
            raise ValueError("Invalid relationship type provided. Use uppercase and underscores.")

        query = (
            f"MATCH (a:Memory {{memory_id: $source_id}}), (b:Memory {{memory_id: $target_id}}) "
            f"MERGE (a)-[r:{safe_relationship_type}]->(b) "
            "RETURN type(r) as rel_type"
        )
        parameters = { "source_id": source_memory_id, "target_id": target_memory_id, }

        async with self.driver.session() as session:
            result = await session.run(query, parameters)
            record = await result.single()
            rel_type = record.data().get('rel_type') if record else None
        
        return rel_type