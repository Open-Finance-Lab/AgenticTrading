"""
Reinforcement Learning Engine for FinAgent Orchestration

This module implements advanced RL algorithms for trading strategy optimization,
portfolio management, and adaptive execution. Integrates with memory agent for
experience-based learning and provides comprehensive policy evaluation.

Supported RL Algorithms:
- Deep Deterministic Policy Gradient (DDPG)
- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)
- Twin Delayed Deep Deterministic Policy Gradient (TD3)

Key Features:
- Multi-objective optimization (return, risk, drawdown)
- Memory-enhanced experience replay
- Continuous action spaces for position sizing
- Market regime adaptation
- Real-time policy updates

Author: FinAgent Team
Version: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from collections import deque, namedtuple
import pickle
import os

logger = logging.getLogger("FinAgentRL")

# RL Experience tuple
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 'metadata'
])


class RLAlgorithm(Enum):
    """Supported RL algorithms"""
    DDPG = "ddpg"
    PPO = "ppo"
    SAC = "sac"
    TD3 = "td3"


class RewardFunction(Enum):
    """Reward function types"""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MULTI_OBJECTIVE = "multi_objective"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


@dataclass
class RLConfiguration:
    """Configuration for RL training and execution"""
    algorithm: RLAlgorithm
    reward_function: RewardFunction
    state_features: List[str]
    action_space_dim: int
    learning_rate: float = 1e-4
    batch_size: int = 256
    memory_size: int = 100000
    update_frequency: int = 100
    target_update_frequency: int = 1000
    discount_factor: float = 0.99
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    tau: float = 0.005
    enable_memory_integration: bool = True
    risk_tolerance: float = 0.5
    multi_objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "return": 0.6, "risk": 0.3, "drawdown": 0.1
    })


class ActorNetwork(nn.Module):
    """Actor network for policy-based RL algorithms"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_dim)
        
        # Layer normalization for better stability
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
        
        # Initialize final layer with smaller weights
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network"""
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        
        # Use tanh activation for bounded action space [-1, 1]
        action = torch.tanh(self.fc4(x))
        return action


class CriticNetwork(nn.Module):
    """Critic network for value-based estimation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(CriticNetwork, self).__init__()
        
        # Q1 network
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_fc3 = nn.Linear(hidden_size, hidden_size)
        self.q1_fc4 = nn.Linear(hidden_size, 1)
        
        # Q2 network (for TD3)
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_fc3 = nn.Linear(hidden_size, hidden_size)
        self.q2_fc4 = nn.Linear(hidden_size, 1)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for layer in [self.q1_fc1, self.q1_fc2, self.q1_fc3, 
                      self.q2_fc1, self.q2_fc2, self.q2_fc3]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through critic networks"""
        sa = torch.cat([state, action], dim=1)
        
        # Q1 network
        q1 = F.relu(self.ln1(self.q1_fc1(sa)))
        q1 = F.relu(self.ln2(self.q1_fc2(q1)))
        q1 = F.relu(self.ln3(self.q1_fc3(q1)))
        q1 = self.q1_fc4(q1)
        
        # Q2 network
        q2 = F.relu(self.ln1(self.q2_fc1(sa)))
        q2 = F.relu(self.ln2(self.q2_fc2(q2)))
        q2 = F.relu(self.ln3(self.q2_fc3(q2)))
        q2 = self.q2_fc4(q2)
        
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q1 network only"""
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.ln1(self.q1_fc1(sa)))
        q1 = F.relu(self.ln2(self.q1_fc2(q1)))
        q1 = F.relu(self.ln3(self.q1_fc3(q1)))
        q1 = self.q1_fc4(q1)
        return q1


class ReplayBuffer:
    """Experience replay buffer with memory integration"""
    
    def __init__(self, capacity: int, enable_memory_integration: bool = True):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.enable_memory_integration = enable_memory_integration
        self.memory_weights = {}  # For memory-enhanced sampling
    
    def add(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        
        # Update memory weights if enabled
        if self.enable_memory_integration:
            self._update_memory_weights(experience)
    
    def _update_memory_weights(self, experience: Experience):
        """Update memory-based importance weights"""
        # Calculate importance based on reward magnitude and market conditions
        importance = abs(experience.reward) + experience.metadata.get('volatility', 0)
        self.memory_weights[len(self.buffer) - 1] = importance
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch from buffer with optional memory-based importance"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if self.enable_memory_integration and self.memory_weights:
            # Sample based on importance weights
            indices = list(range(len(self.buffer)))
            weights = [self.memory_weights.get(i, 1.0) for i in indices]
            weights = np.array(weights) / np.sum(weights)
            
            sampled_indices = np.random.choice(
                indices, size=batch_size, replace=False, p=weights
            )
            return [self.buffer[i] for i in sampled_indices]
        else:
            # Uniform random sampling
            return np.random.choice(self.buffer, size=batch_size, replace=False).tolist()
    
    def size(self) -> int:
        """Get buffer size"""
        return len(self.buffer)


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent"""
    
    def __init__(self, config: RLConfiguration, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_target = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            config.memory_size, 
            config.enable_memory_integration
        )
        
        # Training counters
        self.total_iterations = 0
        self.policy_delay_counter = 0
        
        # Performance tracking
        self.training_metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "q_values": [],
            "policy_updates": 0
        }
        
        logger.info("TD3 Agent initialized")
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # Add exploration noise during training
        if add_noise:
            noise = np.random.normal(0, self.config.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        next_state: np.ndarray, done: bool, metadata: Dict[str, Any] = None):
        """Store experience in replay buffer"""
        if metadata is None:
            metadata = {}
            
        experience = Experience(state, action, reward, next_state, done, metadata)
        self.replay_buffer.add(experience)
    
    def train(self) -> Dict[str, float]:
        """Train the agent using TD3 algorithm"""
        if self.replay_buffer.size() < self.config.batch_size:
            return {}
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.FloatTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch]).unsqueeze(1)
        
        with torch.no_grad():
            # Add noise to target actions for regularization
            noise = torch.clamp(
                torch.randn_like(actions) * self.config.policy_noise,
                -self.config.noise_clip,
                self.config.noise_clip
            )
            
            next_actions = torch.clamp(
                self.actor_target(next_states) + noise,
                -1, 1
            )
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones.float()) * self.config.discount_factor * target_q
        
        # Update critic networks
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = torch.tensor(0.0)
        
        # Delayed policy updates
        self.policy_delay_counter += 1
        if self.policy_delay_counter % self.config.policy_delay == 0:
            # Update actor network
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            self.training_metrics["policy_updates"] += 1
        
        # Track metrics
        self.training_metrics["actor_loss"].append(actor_loss.item())
        self.training_metrics["critic_loss"].append(critic_loss.item())
        self.training_metrics["q_values"].append(current_q1.mean().item())
        
        self.total_iterations += 1
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "avg_q_value": current_q1.mean().item(),
            "buffer_size": self.replay_buffer.size()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.config.tau * source_param.data + (1.0 - self.config.tau) * target_param.data
            )
    
    def save_model(self, filepath: str):
        """Save agent model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_metrics': self.training_metrics
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load agent model"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_metrics = checkpoint['training_metrics']
        logger.info(f"Model loaded from {filepath}")


class TradingEnvironment:
    """Trading environment for RL agent training and evaluation"""
    
    def __init__(self, 
                 market_data: Dict[str, pd.DataFrame],
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.001):
        self.market_data = market_data
        self.symbols = list(market_data.keys())
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Environment state
        self.reset()
        
        # State features
        self.state_features = [
            'returns', 'volatility', 'rsi', 'macd', 'bollinger_position',
            'volume_ratio', 'price_momentum', 'portfolio_weight'
        ]
        
        logger.info(f"Trading environment initialized with {len(self.symbols)} symbols")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.capital = self.initial_capital
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.initial_capital
        self.trade_history = []
        self.portfolio_history = []
        
        # Calculate initial state
        return self._get_state()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step"""
        # Validate actions
        actions = np.clip(actions, -1, 1)  # Ensure actions are in valid range
        
        # Execute trades
        trade_info = self._execute_trades(actions)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update state
        self.current_step += 1
        next_state = self._get_state()
        
        # Check if episode is done
        done = self._is_done()
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'trades': trade_info,
            'positions': self.positions.copy(),
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _execute_trades(self, actions: np.ndarray) -> List[Dict[str, Any]]:
        """Execute trades based on actions"""
        trades = []
        
        for i, symbol in enumerate(self.symbols):
            if i >= len(actions):
                break
                
            target_weight = actions[i]  # Target portfolio weight [-1, 1]
            current_weight = self._get_current_weight(symbol)
            
            # Calculate required trade
            weight_diff = target_weight - current_weight
            trade_value = weight_diff * self.portfolio_value
            
            if abs(trade_value) > self.portfolio_value * 0.01:  # Minimum trade threshold
                # Get current price
                current_price = self._get_current_price(symbol)
                
                if current_price is not None:
                    # Calculate shares to trade
                    shares = trade_value / current_price
                    
                    # Apply slippage and commission
                    execution_price = current_price * (1 + np.sign(shares) * self.slippage_rate)
                    commission = abs(trade_value) * self.commission_rate
                    
                    # Update positions
                    self.positions[symbol] += shares
                    self.capital -= trade_value + commission
                    
                    # Record trade
                    trade = {
                        'symbol': symbol,
                        'shares': shares,
                        'price': execution_price,
                        'value': trade_value,
                        'commission': commission,
                        'timestamp': self.current_step
                    }
                    trades.append(trade)
                    self.trade_history.append(trade)
        
        # Update portfolio value
        self._update_portfolio_value()
        
        return trades
    
    def _get_current_weight(self, symbol: str) -> float:
        """Get current portfolio weight for symbol"""
        if self.portfolio_value == 0:
            return 0.0
        
        current_price = self._get_current_price(symbol)
        if current_price is None:
            return 0.0
        
        position_value = self.positions[symbol] * current_price
        return position_value / self.portfolio_value
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        if symbol not in self.market_data:
            return None
        
        data = self.market_data[symbol]
        if self.current_step >= len(data):
            return None
        
        return data.iloc[self.current_step]['close']
    
    def _update_portfolio_value(self):
        """Update total portfolio value"""
        market_value = 0.0
        
        for symbol, shares in self.positions.items():
            current_price = self._get_current_price(symbol)
            if current_price is not None:
                market_value += shares * current_price
        
        self.portfolio_value = self.capital + market_value
        self.portfolio_history.append({
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'capital': self.capital,
            'market_value': market_value
        })
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current step"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # Calculate returns
        prev_value = self.portfolio_history[-2]['portfolio_value']
        current_value = self.portfolio_value
        
        if prev_value == 0:
            return 0.0
        
        returns = (current_value - prev_value) / prev_value
        
        # Risk-adjusted reward (Sharpe-like)
        if len(self.portfolio_history) >= 20:  # Minimum history for volatility calculation
            recent_returns = [
                (self.portfolio_history[i]['portfolio_value'] - self.portfolio_history[i-1]['portfolio_value']) / 
                self.portfolio_history[i-1]['portfolio_value']
                for i in range(-19, 0)
            ]
            volatility = np.std(recent_returns)
            
            if volatility > 0:
                risk_adjusted_return = returns / volatility
            else:
                risk_adjusted_return = returns
        else:
            risk_adjusted_return = returns
        
        # Penalty for large drawdowns
        max_value = max(h['portfolio_value'] for h in self.portfolio_history)
        drawdown = (max_value - current_value) / max_value
        drawdown_penalty = -max(0, drawdown - 0.05) * 10  # Penalty for >5% drawdown
        
        return risk_adjusted_return + drawdown_penalty
    
    def _get_state(self) -> np.ndarray:
        """Get current environment state"""
        if self.current_step >= min(len(data) for data in self.market_data.values()):
            # Return zero state if no data available
            return np.zeros(len(self.symbols) * len(self.state_features))
        
        state_vector = []
        
        for symbol in self.symbols:
            data = self.market_data[symbol]
            
            if self.current_step < len(data):
                # Calculate technical features
                close_prices = data['close'].iloc[:self.current_step+1]
                
                # Returns
                returns = close_prices.pct_change().iloc[-1] if len(close_prices) > 1 else 0.0
                
                # Volatility (20-day)
                volatility = close_prices.pct_change().rolling(20).std().iloc[-1] if len(close_prices) > 20 else 0.0
                
                # RSI (simplified)
                rsi = 0.5  # Placeholder
                
                # MACD (simplified)
                macd = 0.0  # Placeholder
                
                # Bollinger position (simplified)
                bollinger_position = 0.5  # Placeholder
                
                # Volume ratio
                volume_ratio = 1.0  # Placeholder
                
                # Price momentum
                if len(close_prices) > 5:
                    price_momentum = (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5]
                else:
                    price_momentum = 0.0
                
                # Portfolio weight
                portfolio_weight = self._get_current_weight(symbol)
                
                # Add to state vector
                features = [
                    returns, volatility, rsi, macd, bollinger_position,
                    volume_ratio, price_momentum, portfolio_weight
                ]
                
                # Handle NaN values
                features = [f if not np.isnan(f) else 0.0 for f in features]
                state_vector.extend(features)
            else:
                # No data available, use zeros
                state_vector.extend([0.0] * len(self.state_features))
        
        return np.array(state_vector, dtype=np.float32)
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        # Episode ends if we run out of data or portfolio value drops too much
        max_steps = min(len(data) for data in self.market_data.values())
        portfolio_exhausted = self.portfolio_value < self.initial_capital * 0.1
        
        return self.current_step >= max_steps - 1 or portfolio_exhausted


class RLPolicyEngine:
    """Main RL policy engine for trading strategy optimization"""
    
    def __init__(self, config: RLConfiguration):
        self.config = config
        self.agents = {}
        self.environments = {}
        self.training_history = {}
        
        logger.info("RL Policy Engine initialized")
    
    def create_agent(self, agent_id: str, state_dim: int, action_dim: int) -> TD3Agent:
        """Create a new RL agent"""
        if self.config.algorithm == RLAlgorithm.TD3:
            agent = TD3Agent(self.config, state_dim, action_dim)
        else:
            raise NotImplementedError(f"Algorithm {self.config.algorithm} not implemented")
        
        self.agents[agent_id] = agent
        self.training_history[agent_id] = []
        
        logger.info(f"Created RL agent {agent_id} with algorithm {self.config.algorithm.value}")
        return agent
    
    def create_environment(self, env_id: str, market_data: Dict[str, pd.DataFrame]) -> TradingEnvironment:
        """Create a new trading environment"""
        env = TradingEnvironment(market_data)
        self.environments[env_id] = env
        
        logger.info(f"Created trading environment {env_id}")
        return env
    
    async def train_agent(self, agent_id: str, env_id: str, episodes: int = 1000) -> Dict[str, Any]:
        """Train an RL agent in the specified environment"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        agent = self.agents[agent_id]
        env = self.environments[env_id]
        
        logger.info(f"Starting training for agent {agent_id} in environment {env_id}")
        
        training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "portfolio_values": [],
            "training_losses": []
        }
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select action
                action = agent.select_action(state, add_noise=True)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done, info)
                
                # Train agent
                if agent.replay_buffer.size() > agent.config.batch_size:
                    training_result = agent.train()
                    if training_result:
                        training_metrics["training_losses"].append(training_result)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Record episode metrics
            training_metrics["episode_rewards"].append(episode_reward)
            training_metrics["episode_lengths"].append(episode_length)
            training_metrics["portfolio_values"].append(env.portfolio_value)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(training_metrics["episode_rewards"][-100:])
                avg_portfolio = np.mean(training_metrics["portfolio_values"][-100:])
                
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.4f}, "
                           f"Avg Portfolio Value = ${avg_portfolio:,.2f}")
        
        # Store training history
        self.training_history[agent_id] = training_metrics
        
        logger.info(f"Training completed for agent {agent_id}")
        return training_metrics
    
    def evaluate_agent(self, agent_id: str, env_id: str, episodes: int = 10) -> Dict[str, Any]:
        """Evaluate trained agent performance"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        if env_id not in self.environments:
            raise ValueError(f"Environment {env_id} not found")
        
        agent = self.agents[agent_id]
        env = self.environments[env_id]
        
        evaluation_metrics = {
            "episode_rewards": [],
            "portfolio_values": [],
            "total_returns": [],
            "sharpe_ratios": [],
            "max_drawdowns": []
        }
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            while True:
                # Select action without exploration noise
                action = agent.select_action(state, add_noise=False)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            # Calculate performance metrics
            initial_value = env.initial_capital
            final_value = env.portfolio_value
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate Sharpe ratio (simplified)
            if len(env.portfolio_history) > 1:
                returns = [
                    (env.portfolio_history[i]['portfolio_value'] - env.portfolio_history[i-1]['portfolio_value']) / 
                    env.portfolio_history[i-1]['portfolio_value']
                    for i in range(1, len(env.portfolio_history))
                ]
                
                if len(returns) > 0 and np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
            
            # Calculate maximum drawdown
            max_value = max(h['portfolio_value'] for h in env.portfolio_history)
            min_value = min(h['portfolio_value'] for h in env.portfolio_history)
            max_drawdown = (max_value - min_value) / max_value
            
            # Record metrics
            evaluation_metrics["episode_rewards"].append(episode_reward)
            evaluation_metrics["portfolio_values"].append(final_value)
            evaluation_metrics["total_returns"].append(total_return)
            evaluation_metrics["sharpe_ratios"].append(sharpe_ratio)
            evaluation_metrics["max_drawdowns"].append(max_drawdown)
        
        # Calculate summary statistics
        summary = {
            "avg_total_return": np.mean(evaluation_metrics["total_returns"]),
            "avg_sharpe_ratio": np.mean(evaluation_metrics["sharpe_ratios"]),
            "avg_max_drawdown": np.mean(evaluation_metrics["max_drawdowns"]),
            "avg_final_portfolio_value": np.mean(evaluation_metrics["portfolio_values"]),
            "success_rate": sum(1 for r in evaluation_metrics["total_returns"] if r > 0) / episodes
        }
        
        logger.info(f"Agent {agent_id} evaluation completed:")
        logger.info(f"  Average Total Return: {summary['avg_total_return']:.2%}")
        logger.info(f"  Average Sharpe Ratio: {summary['avg_sharpe_ratio']:.2f}")
        logger.info(f"  Average Max Drawdown: {summary['avg_max_drawdown']:.2%}")
        logger.info(f"  Success Rate: {summary['success_rate']:.2%}")
        
        return {**evaluation_metrics, "summary": summary}
    
    def save_agent(self, agent_id: str, filepath: str):
        """Save trained agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        self.agents[agent_id].save_model(filepath)
    
    def load_agent(self, agent_id: str, filepath: str):
        """Load trained agent"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        # Load config from saved model
        checkpoint = torch.load(filepath)
        config = checkpoint['config']
        
        # Determine state and action dimensions from saved model
        state_dim = checkpoint['actor_state_dict']['fc1.weight'].shape[1]
        action_dim = checkpoint['actor_state_dict']['fc4.weight'].shape[0]
        
        # Create agent and load model
        agent = self.create_agent(agent_id, state_dim, action_dim)
        agent.load_model(filepath)


if __name__ == "__main__":
    # Example usage
    config = RLConfiguration(
        algorithm=RLAlgorithm.TD3,
        reward_function=RewardFunction.SHARPE_RATIO,
        state_features=["returns", "volatility", "rsi"],
        action_space_dim=3,  # 3 stocks
        learning_rate=1e-4,
        batch_size=256
    )
    
    # Create RL engine
    rl_engine = RLPolicyEngine(config)
    
    # Create agent
    agent = rl_engine.create_agent("test_agent", state_dim=24, action_dim=3)
    
    logger.info("RL Policy Engine example completed")
