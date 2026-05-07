# Trust Duel: Person vs Agent

A small reinforcement learning inspired game based on the **Iterated Prisoner's Dilemma**.

The project compares fixed game-theory strategies with a simple Q-learning agent in a repeated decision-making environment.

## Project Goal

The goal is to demonstrate core reinforcement learning concepts through a playable and visual game:

- Agent
- Environment
- State
- Action
- Reward
- Policy
- Q-table
- Exploration vs exploitation
- Cumulative reward

## Game Rules

Each player chooses one action per round:

- `C` = Cooperate
- `D` = Defect

Payoff matrix:

| Player A / Player B | Cooperate | Defect |
|---|---:|---:|
| Cooperate | 3, 3 | 0, 5 |
| Defect | 5, 0 | 1, 1 |

## Project Structure

```text
trust-duel-rl-game/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── 01_trust_duel_game.ipynb
├── src/
│   ├── __init__.py
│   ├── game.py
│   ├── strategies.py
│   └── q_agent.py
└── reports/
    └── figures/
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open:

```text
notebooks/01_trust_duel_game.ipynb
```

## Main Notebook Sections

1. Project Introduction  
2. Game Rules  
3. Payoff Matrix  
4. Core Game Engine  
5. Fixed Strategy Agents  
6. Strategy Tournament  
7. Results Visualization  
8. Human vs Agent Game  
9. Q-Learning Agent  
10. Training and Evaluation  
11. Final Interpretation  

## Key Insight

There is no universally best strategy.  
A strategy performs well depending on the environment and the opponents it faces.

In repeated games, good performance often requires a balance between:

- cooperation,
- retaliation,
- forgiveness,
- clarity,
- adaptability.
