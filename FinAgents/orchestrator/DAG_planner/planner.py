from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import networkx as nx
from abc import ABC, abstractmethod

@dataclass
class TaskNode:
    """Represents a task node in the DAG"""
    task_id: str
    agent_type: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    status: str = "pending"

class DAGPlannerAgent(ABC):
    """Abstract base class for DAG Planner Agent
    
    Responsible for converting high-level strategic queries into executable DAG task flows.
    Implements task decomposition, dependency analysis, and graph construction functionality.
    """
    
    def __init__(self):
        self.dag = nx.DiGraph()
        self.task_registry: Dict[str, TaskNode] = {}
        
    @abstractmethod
    async def plan(self, query: str) -> nx.DiGraph:
        """Convert user query into a DAG
        
        Args:
            query: High-level strategic query from user
            
        Returns:
            nx.DiGraph: Constructed task DAG
        """
        pass
    
    def add_task(self, task: TaskNode) -> None:
        """Add a task node to the DAG
        
        Args:
            task: Task node to be added
        """
        self.task_registry[task.task_id] = task
        self.dag.add_node(task.task_id, **task.__dict__)
        
        # Add dependency edges
        for dep in task.dependencies:
            if dep in self.task_registry:
                self.dag.add_edge(dep, task.task_id)
    
    def validate_dag(self) -> bool:
        """Validate the DAG's validity
        
        Returns:
            bool: Whether the DAG is valid
        """
        return nx.is_directed_acyclic_graph(self.dag)
    
    def get_execution_order(self) -> List[str]:
        """Get the execution order of tasks
        
        Returns:
            List[str]: List of task IDs in topological order
        """
        return list(nx.topological_sort(self.dag))
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get task status
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[str]: Task status, None if task doesn't exist
        """
        task = self.task_registry.get(task_id)
        return task.status if task else None
    
    def update_task_status(self, task_id: str, status: str) -> None:
        """Update task status
        
        Args:
            task_id: Task ID
            status: New status
        """
        if task_id in self.task_registry:
            self.task_registry[task_id].status = status
            self.dag.nodes[task_id]['status'] = status 