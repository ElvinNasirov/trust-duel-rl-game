"""
Q-learning agent for Trust Duel.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .game import ACTIONS, get_payoff


State = Tuple[str, str]


def get_state(my_history, opponent_history) -> State:
    """
    Build a simple state from the latest actions.

    State = (my_previous_action, opponent_previous_action)

    If there is no previous action, use START.
    """
    my_last = my_history[-1] if len(my_history) > 0 else "START"
    opponent_last = opponent_history[-1] if len(opponent_history) > 0 else "START"
    return (my_last, opponent_last)


class QLearningAgent:
    """
    Tabular Q-learning agent.

    The agent learns Q(state, action), where:
    - state = previous actions
    - action = cooperate or defect
    - reward = payoff from the game
    """

    name = "Q-Learning Agent"

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.q_table: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(len(ACTIONS)))

    def reset(self) -> None:
        """No episode-specific internal state is needed."""
        return None

    def choose_action(self, state: State, training: bool = True) -> str:
        """Choose action using epsilon-greedy policy."""
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS).item()

        q_values = self.q_table[state]
        best_action_idx = int(np.argmax(q_values))
        return ACTIONS[best_action_idx]

    def update(self, state: State, action: str, reward: float, next_state: State) -> None:
        """Apply Q-learning update."""
        action_idx = ACTIONS.index(action)

        old_value = self.q_table[state][action_idx]
        next_best_value = float(np.max(self.q_table[next_state]))

        target = reward + self.gamma * next_best_value
        self.q_table[state][action_idx] = old_value + self.alpha * (target - old_value)

    def decay_epsilon(self) -> None:
        """Reduce exploration over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def q_table_dataframe(self) -> pd.DataFrame:
        """Return Q-table as a clean DataFrame."""
        rows = []
        for state, values in self.q_table.items():
            rows.append(
                {
                    "state_my_previous": state[0],
                    "state_opponent_previous": state[1],
                    "Q_C": values[0],
                    "Q_D": values[1],
                    "best_action": ACTIONS[int(np.argmax(values))],
                }
            )
        return pd.DataFrame(rows)


class TrainedQAgentStrategy:
    """
    Wrapper that lets a trained QLearningAgent behave like a fixed strategy.
    """

    name = "Trained Q-Agent"

    def __init__(self, agent: QLearningAgent):
        self.agent = agent

    def reset(self) -> None:
        return None

    def choose_action(self, my_history, opponent_history) -> str:
        state = get_state(my_history, opponent_history)
        return self.agent.choose_action(state, training=False)


def train_q_agent(
    opponent_strategy,
    episodes: int = 1000,
    rounds_per_episode: int = 100,
    agent: QLearningAgent | None = None,
    seed: int | None = 42,
) -> tuple[QLearningAgent, pd.DataFrame]:
    """
    Train a Q-learning agent against one fixed opponent strategy.

    Returns
    -------
    agent:
        Trained QLearningAgent.

    training_log:
        Episode-level rewards and epsilon values.
    """
    rng = np.random.default_rng(seed)

    if agent is None:
        agent = QLearningAgent(seed=seed)

    episode_rows = []

    for episode in range(1, episodes + 1):
        opponent = copy.deepcopy(opponent_strategy)

        if hasattr(opponent, "reset"):
            opponent.reset()

        agent_history = []
        opponent_history = []
        episode_reward = 0

        for _ in range(rounds_per_episode):
            state = get_state(agent_history, opponent_history)

            agent_action = agent.choose_action(state, training=True)
            opponent_action = opponent.choose_action(opponent_history, agent_history)

            reward_agent, _ = get_payoff(agent_action, opponent_action)

            agent_history.append(agent_action)
            opponent_history.append(opponent_action)

            next_state = get_state(agent_history, opponent_history)
            agent.update(state, agent_action, reward_agent, next_state)

            episode_reward += reward_agent

        agent.decay_epsilon()

        episode_rows.append(
            {
                "episode": episode,
                "total_reward": episode_reward,
                "epsilon": agent.epsilon,
            }
        )

    return agent, pd.DataFrame(episode_rows)


def train_q_agent_against_pool(
    opponent_strategies,
    episodes: int = 1500,
    rounds_per_episode: int = 100,
    seed: int | None = 42,
) -> tuple[QLearningAgent, pd.DataFrame]:
    """
    Train a Q-learning agent against a balanced pool of opponents.
    Each strategy appears roughly the same number of times.
    """
    rng = np.random.default_rng(seed)
    agent = QLearningAgent(seed=seed)

    all_logs = []

    strategy_indices = []
    while len(strategy_indices) < episodes:
        cycle = list(range(len(opponent_strategies)))
        rng.shuffle(cycle)
        strategy_indices.extend(cycle)

    strategy_indices = strategy_indices[:episodes]

    for episode, idx in enumerate(strategy_indices, start=1):
        opponent_template = copy.deepcopy(opponent_strategies[idx])

        agent, log = train_q_agent(
            opponent_template,
            episodes=1,
            rounds_per_episode=rounds_per_episode,
            agent=agent,
            seed=int(rng.integers(0, 1_000_000)),
        )

        row = log.iloc[0].to_dict()
        row["episode"] = episode
        row["opponent"] = opponent_template.name
        row["avg_reward_per_round"] = row["total_reward"] / rounds_per_episode
        all_logs.append(row)

    return agent, pd.DataFrame(all_logs)
