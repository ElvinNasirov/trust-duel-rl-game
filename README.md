# Trust Duel: Person vs Agent

**Trust Duel** is a reinforcement learning inspired game based on the Iterated Prisoner's Dilemma.

The project compares fixed game-theory strategies with a simple Q-learning agent. The goal is to show how an agent can make decisions through repeated interaction, rewards, exploration, and learning.

---

## Project Goal

The main goal of this project is to demonstrate core Reinforcement Learning concepts through a simple but meaningful game.

Instead of building a complex visual arcade game, this project focuses on decision-making:

- Should the agent cooperate?
- Should the agent defect?
- How does the agent react to different opponents?
- Can the agent learn from rewards over repeated rounds?

The project contains two main parts:

1. **Algorithm vs Algorithm**  
   Fixed strategies play against each other in a tournament.

2. **Person vs Agent / Agent Training**  
   A Q-learning agent learns to play against different strategies.

---

## Game Background

The game is based on the **Iterated Prisoner's Dilemma**.

Each round, both players choose one of two actions:

- `C` = Cooperate
- `D` = Defect

The reward depends on both players' actions.

| Player A / Player B | B Cooperates |  B Defects |
| ------------------- | -----------: | ---------: |
| A Cooperates        |   A: 3, B: 3 | A: 0, B: 5 |
| A Defects           |   A: 5, B: 0 | A: 1, B: 1 |

The dilemma:

- Mutual cooperation gives both players a good reward.
- Defection can exploit a cooperative opponent.
- Mutual defection gives both players a low reward.
- In repeated games, trust, punishment, forgiveness, and adaptation become important.

---

## Reinforcement Learning Concepts

This project maps the game to Reinforcement Learning concepts:

| RL Concept   | Meaning in This Project                        |
| ------------ | ---------------------------------------------- |
| Agent        | The strategy or Q-learning player              |
| Environment  | The repeated Prisoner's Dilemma game           |
| State        | Previous actions of both players               |
| Action       | Cooperate (`C`) or Defect (`D`)                |
| Reward       | Coins received from the payoff matrix          |
| Episode      | One full repeated match                        |
| Policy       | The rule used to choose actions                |
| Q-table      | Learned values for actions in different states |
| Exploration  | Trying random actions                          |
| Exploitation | Choosing the best-known action                 |

---

## Implemented Fixed Strategies

The project includes several fixed strategies.

| Strategy               | Rule                                                              | Behavior                                       |
| ---------------------- | ----------------------------------------------------------------- | ---------------------------------------------- |
| Always Cooperate       | Always plays `C`                                                  | Friendly but easy to exploit                   |
| Always Defect          | Always plays `D`                                                  | Aggressive but often creates low mutual reward |
| Random                 | Randomly chooses `C` or `D`                                       | Unpredictable baseline                         |
| Tit for Tat            | Starts with `C`, then copies the opponent's previous move         | Clear, reciprocal, strong classic strategy     |
| Grim Trigger           | Starts with `C`, but defects forever after one opponent defection | Strict and unforgiving                         |
| Tit for Two Tats       | Defects only after two consecutive opponent defections            | More tolerant than Tit for Tat                 |
| Forgiving Tit for Tat  | Copies defection but sometimes forgives                           | More robust in noisy environments              |
| Suspicious Tit for Tat | Starts with `D`, then copies the opponent's previous move         | Defensive variant                              |

---

## Q-Learning Agent

The project also includes a simple tabular Q-learning agent.

The Q-agent learns values for `Q(state, action)`.

The state is based on the previous actions:

```text
state = (my_previous_action, opponent_previous_action)
```

Possible actions:

```text
C = Cooperate
D = Defect
```

Default Q-learning parameters:

| Parameter       | Value | Meaning                           |
| --------------- | ----: | --------------------------------- |
| `alpha`         |   0.1 | Learning rate                     |
| `gamma`         |  0.95 | Future reward importance          |
| `epsilon`       |   1.0 | Initial exploration rate          |
| `epsilon_min`   |  0.05 | Minimum exploration rate          |
| `epsilon_decay` | 0.995 | How quickly exploration decreases |

The agent uses an **epsilon-greedy policy**:

- With probability `epsilon`, it explores by choosing a random action.
- With probability `1 - epsilon`, it exploits the best action from the Q-table.

After each episode, epsilon decreases:

```text
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

This means the agent starts by exploring a lot, then gradually relies more on learned Q-values.

---

## Training Modes

The notebook demonstrates two training modes.

### 1. Training Against One Opponent

The Q-agent is trained against a single fixed strategy, such as Tit for Tat.

This is useful for understanding how the agent learns in a stable environment.

### 2. Training Against a Balanced Pool of Strategies

The Q-agent is also trained against a pool of different opponents.

In this setup, the opponent changes between episodes. The balanced pool prevents the agent from overfitting to one strategy and makes the training more general.

Example pool:

- Always Cooperate
- Always Defect
- Random
- Tit for Tat
- Grim Trigger
- Forgiving Tit for Tat

---

## Visual Outputs

The notebook generates several visual outputs:

| Visualization              | Purpose                                                    |
| -------------------------- | ---------------------------------------------------------- |
| Payoff matrix              | Shows the rules of the game                                |
| Strategy guide table       | Explains how each fixed strategy behaves                   |
| Tournament ranking chart   | Compares fixed strategies                                  |
| Cumulative reward chart    | Shows how rewards accumulate over repeated rounds          |
| Epsilon decay chart        | Shows how exploration decreases during training            |
| Average reward by opponent | Shows which opponents are easier or harder for the Q-agent |
| Q-table heatmap            | Shows what the Q-agent learned                             |

Generated figures are saved in:

```text
reports/figures/
```

---

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
    ├── README.md
    └── figures/
        └── .gitkeep
```

---

## File Descriptions

| File                                 | Purpose                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------ |
| `notebooks/01_trust_duel_game.ipynb` | Main notebook with explanation, experiments, visualizations, and results |
| `src/game.py`                        | Core game engine, payoff matrix, match simulation, tournament logic      |
| `src/strategies.py`                  | Fixed strategies used in the tournament                                  |
| `src/q_agent.py`                     | Q-learning agent and training functions                                  |
| `reports/figures/`                   | Saved charts generated by the notebook                                   |
| `requirements.txt`                   | Python dependencies                                                      |
| `.gitignore`                         | Files and folders ignored by Git                                         |

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/trust-duel-rl-game.git
cd trust-duel-rl-game
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/01_trust_duel_game.ipynb
```

Then run the notebook cells from top to bottom.

---

## Main Results

The fixed-strategy tournament shows that simple strategies can perform surprisingly well.

Strategies such as **Tit for Tat** and **Forgiving Tit for Tat** are strong because they combine:

- cooperation,
- retaliation,
- forgiveness,
- clarity.

The Q-learning agent shows how an agent can learn from rewards instead of following a manually written strategy.

However, there is no universally best strategy. Performance depends on the environment and the opponents.

---

## Key Takeaway

This project demonstrates that Reinforcement Learning is not only about complex environments or games with graphics.

At its core, RL is about repeated decision-making:

```text
state -> action -> reward -> updated knowledge -> better future decisions
```

The Trust Duel game shows how cooperation, betrayal, punishment, forgiveness, and learning can be modeled with simple code and clear visualizations.
