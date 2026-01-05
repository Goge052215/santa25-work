## Essentials

Listing ALL essential equations for the Hybrid-NFL problems covered in [Fang et al. (2023)](hybrid-RF-NFP.pdf)

### Problem Statement

The mathematical model for 2D irregular-piece packing problems can be expressed as follows: Given a plate $P$ of width $W$, a set group of pieces can be arranged with quantity $n$ as $\{P_1, P_2, \ldots, P_n\}$. Each piece $P_i$ has a width $w_i$ and a height $h_i$. The piece number follows the ordered natural sequence; the objective function of piece packing and the constraints of packing optimization are shown in Formulas (1) and (2), respectively:
$$
    \max_{\rho} = \frac{\sum_{i=1}^n s_i}{WH} \tag{1}
$$

$$
    \text{s.t.} \, \left\{ 
        \begin{aligned}
            &P_i \in P \\
            &P_i \cap P_j = \emptyset \\
            &i \neq j,\, i,j \in \{1,2,\dots,n\}
        \end{aligned}
    \right\} \tag{2}
$$

where $S_i$ is the area of the $i$-th piece, and $H$ is the plate height occupied by the pieces after packing. Here, the maximum utilization rate $\rho$ of the optimization target is equivalent to the minimum total height $H$ of the packing.

### Preliminaries

1. No-Fit-Polygon (NFP): A polygon $P$ is called a no-fit polygon if it does not fit into the plate $P$ without rotation.
2. BL-Positioning Algorithm: We calculate the placement of 2D irregular pieces using the classic BL positioning algorithm combined with NFP.

### Squeeze Optimization

#### Reward-Model

The packing sequence optimization based on hybrid RL can be modeled as a multistage decision process:
$$
    s_0,a_1,s_1,r_1,a_2,s_2,r_2,\ldots,a_n,s_n,r_n \quad \pi(s|a)
$$

Here, $s_0$ represents the state when pieces are not arranged, $a_i$ and $s_i$ are actions and states of the $i$-th stage, respectively, $i \in [1, n]$, and $r_i$ is the immediate reward the piece obtains in state $s_i$ at the $i$-th stage. $\pi(s|a)$ is a specific evaluation strategy (e.g. greedy algorith,).

#### Monte-Carlo Reinforcement Learning (MCRL)

Generally, the value of a state is equal to the average of all rewards calculated using the state in multiple episodes. When the current strategy of the agent is to be evaluated, many episodes can be generated using the strategy $\pi(s|a)$.

Then, the discount–reward–return value at state s in each episode can be calculated as shown in Formula (3). The average reward value can be calculated with two methods: the first visit or every visit. The first visit means that when calculating the value function at state $s$, only the value returned when state $s$ is visited for the ﬁrst time in each episode is used, as shown in Formula (4). 

While calculating the value function at state s, the return value of all visits at state $s$ is utilized, called every visit, as shown in Formula (5). According to the characteristics of piece sequences, we used the ﬁrst visit method to calculate the value function at state $s$ and replaced the value function with the average reward value through different episodes:
$$
    R_i(s) = r_i + \gamma r_{i+1} + \cdots + \gamma^{n-1}r_n \tag{3}
$$

$$
    Q(s) = \frac{R_{11}(s) + R_{21}(s) + \cdots + R_{i1}(s)}{N(s)} \tag{4}
$$

$$
    Q(s) = \frac{R_{11}(s) + R_{12}(s) + \cdots + R_{21}(s) + \cdots}{N(s)} \tag{5}
$$

where $s$ is the state, $r_i$ is the immediate reward of the i-th stage, $\gamma$ is the discount factor, which represents how much the future reward can be observed in the current state, and $R_i(s)$ is the return value of the discounted reward at state $s$ in the i-th episode. $Q(s)$ is the average reward value at state $s$, which can help the agent select the next possible action $a$ from the current state $s$. 

In other words, the next piece with a smaller packing height can be selected according to the corresponding state information. A current optimal sequence can be obtained according to the continuous update of $Q(s,a)$, represented by $S_{\text{opt}}$. The expression of $Q(s,a)$ is displayed in Formula (6), where $N(s)$ represents the number of the same state–action pairs appearing in multiple episodes:
$$
    Q(s,a) = Q(s,a) + \frac{1}{N(s,a)} (R-Q(s,a)) \tag{6}
$$

The exploratory MC reinforcement learning (EMCRL) method (that is, the policy $\pi$ is deﬁned as each trial starting from a random initial state to the termination state) and the on-policy MC reinforcement learning (OMCRL) method (that is, the policy $\pi$ is deﬁned as using the $\epsilon$-greedy algorithm for strategy improvement, as shown in Formula (7), where $|A(s)|$ is the number of states, and $\epsilon$ is the probability of random exploration) are adopted to optimize the 2D irregular-piece packing sequence:
$$
    \pi(a|s) \leftarrow \begin{cases}
        1 - \epsilon + \frac{\epsilon}{|A(s)|}, &\text{if } a = \arg\max_{a} Q(s,a) \\
        \frac{\epsilon}{|A(s)|}, &\text{if } a \neq \arg\max_{a} Q(s,a) \\
    \end{cases} \tag{7}
$$

The total number of episodes is set to $m$. After each episode, the state-sequence set and action-sequence set of pieces change with the change of the average return value of the reward accumulation, which promotes a change in the piece sequence, and further promotes a change in packing height and raw materials utilization. Therefore, the current optimal sequence $S_{\text{opt}}$ represents a solution to the 2D irregular-piece packing problem. Two-dimensional irregular-piece packing based on MCRL is shown in Algorithm 1.

We first initialize the constants and states:
```cpp
using PieceIndex = int;
using State = int;   // Index of the last placed piece
using Action = int;  // Action is selecting the next piece index

// Constants
const int START_STATE = -1;
const double C = 100.0;      // Reward constant
const double EPSILON = 0.1;  // Exploration rate for epsilon-greedy

double calculate_packing_height(const std::vector<PieceIndex>& sequence) {
    // Implementation of BL/NFP placement strategy would go here
    // Returning a dummy random height for demonstration
    return 100.0 + (rand() % 20); 
}
```

**Algorithm 1:** MCRL for a 2D irregular-piece packing problem
```cpp
// Algorithm 1: MCRL for 2D Irregular Packing
std::vector<PieceIndex> mcrl_packing_optimization(int n_pieces, int m_episodes) {
    // Initialize Q-table: Q[state][action] -> value
    // State is the previous piece, Action is the next piece
    std::map<State, std::map<Action, double>> Q;
    
    // Returns list: Returns[state][action] -> list of returns
    std::map<State, std::map<Action, std::vector<double>>> returns_map;
    
    std::vector<PieceIndex> s_opt;
    double min_height_opt = std::numeric_limits<double>::max();

    std::mt19937 rng(42);
    
    // Main Loop: t = 1 to m
    for (int t = 0; t < m_episodes; ++t) {
        // --- 1. Generate an Episode ---
        std::vector<std::pair<State, Action>> episode_trajectory;
        std::vector<PieceIndex> current_sequence;
        std::unordered_set<PieceIndex> remaining_pieces;
        
        for(int i=0; i<n_pieces; ++i) remaining_pieces.insert(i);
        
        State current_state = START_STATE;
        // Construct the sequence piece by piece
        for (int step = 0; step < n_pieces; ++step) {
            Action selected_action;
            
            // Epsilon-greedy policy \pi(a|s)
            std::uniform_real_distribution<> dist(0.0, 1.0);
            bool explore = (dist(rng) < EPSILON);
            
            if (explore || Q[current_state].empty()) {
                // Random selection from remaining
                auto it = remaining_pieces.begin();
                std::advance(it, rng() % remaining_pieces.size());
                selected_action = *it;
            } else {
                // Greedy selection: argmax Q(s, a) among remaining pieces
                double max_q = -std::numeric_limits<double>::max();
                Action best_a = -1;
                bool found_in_q = false;
                
                for (PieceIndex candidate : remaining_pieces) {
                    if (Q[current_state].count(candidate)) {
                        if (Q[current_state][candidate] > max_q) {
                            max_q = Q[current_state][candidate];
                            best_a = candidate;
                            found_in_q = true;
                        }
                    }
                }
                
                if (found_in_q) {
                    selected_action = best_a;
                } else {
                    // Fallback if no history for valid next actions
                    auto it = remaining_pieces.begin();
                    std::advance(it, rng() % remaining_pieces.size());
                    selected_action = *it;
                }
            }
            
            // Execute action
            current_sequence.push_back(selected_action);
            episode_trajectory.push_back({current_state, selected_action});
            remaining_pieces.erase(selected_action);
            
            // Update state: s_i = a_i (as per paper definition)
            current_state = selected_action; 
        }

        // --- 2. Calculate Reward ---
        // Evaluate the complete sequence using positioning strategy (BL)
        double H = calculate_packing_height(current_sequence);
        double reward = (H > 0) ? (C / H) : 0.0;
        
        // Update Optimal Solution S_opt
        if (H < min_height_opt) {
            min_height_opt = H;
            s_opt = current_sequence;
            std::cout << "Episode " << t << ": New Best Height = " << min_height_opt << std::endl;
        }
        
        // --- 3. Update Q-Table (First-visit MC) ---
        std::set<std::pair<State, Action>> visited_pairs;
        
        for (const auto& step : episode_trajectory) {
            State s = step.first;
            Action a = step.second;
            
            // Check first visit in this episode
            if (visited_pairs.find({s, a}) == visited_pairs.end()) {
                visited_pairs.insert({s, a});
                
                // Append return R (which is just 'reward' here as gamma=1 and intermediate r=0)
                returns_map[s][a].push_back(reward);
                
                // Q(s, a) <- average(Returns(s, a))
                const auto& R_list = returns_map[s][a];
                double sum = 0.0;
                for (double val : R_list) sum += val;
                Q[s][a] = sum / R_list.size();
            }
        }
    }

    return s_opt;
}
```

#### Q-Learning and Sarsa-Learning

Q-learning and Sarsa-learning are important components of classic RL, both of which belong to temporal-difference learning (TD learning). To avoid the current best action falling into a local optimum, a certain probability of exploration in generating state–action pairs is used, and the $\epsilon$-greedy algorithm is set as $\pi$ strategy.

Q-learning and Sarsa-learning are updated through continuous interaction with the environment, and the agent automatically learns the action strategy of each step to accumulate the maximum reward. The long-term cumulative reward is represented by the $Q(s,a)$ value table, which guides the packing sequence of the next piece. The updates of $Q(s,a)$ for Q-learning and Sarsa-learning are shown in Formulas (8) and (9), respectively:
$$
    Q(s,a) = Q(s,a) + \alpha \Bigl[r(s,a) + \gamma * \max\left(Q(s',a')\right) - Q(s,a)\Bigr] \tag{8}
$$

$$
    Q(s,a) = Q(s,a) + \alpha \Bigl[r(s,a) + \gamma * Q(s',a')\Bigr] \tag{9}
$$

where $\alpha$ is the learning rate, $\gamma$ is the discounted factor, and $s'$ and $a'$ represent the state and action of the next stage, respectively.

**Algorithm 2:** Q-learning for a 2D irregular packing problem
```cpp
// Algorithm 2: Q-learning for 2D Irregular Packing
std::vector<PieceIndex> q_learning_packing_optimization(int n_pieces, int m_episodes) {
    // Initialize Q-table: Q[state][action] -> value
    std::map<State, std::map<Action, double>> Q;
    
    std::vector<PieceIndex> s_opt;
    double min_height_opt = std::numeric_limits<double>::max();
    
    std::mt19937 rng(42);
    
    // Hyperparameters
    double alpha = 0.5; // Learning rate
    double gamma = 1.0; // Discount factor
    
    for (int t = 0; t < m_episodes; ++t) {
        std::vector<PieceIndex> current_sequence;
        std::unordered_set<PieceIndex> remaining_pieces;
        for(int i=0; i<n_pieces; ++i) remaining_pieces.insert(i);
        
        State s_prev = START_STATE; // s_{i-1}
        
        for (int i = 0; i < n_pieces; ++i) {
            Action a_i; // a_i
            
            // --- Choose a_i at s_{i-1} using epsilon-greedy ---
            std::uniform_real_distribution<> dist(0.0, 1.0);
            bool explore = (dist(rng) < EPSILON);
            
            if (explore || Q[s_prev].empty()) {
                auto it = remaining_pieces.begin();
                std::advance(it, rng() % remaining_pieces.size());
                a_i = *it;
            } else {
                // Greedy w.r.t Q
                double max_q = -std::numeric_limits<double>::max();
                Action best_a = -1;
                bool found = false;
                for (PieceIndex candidate : remaining_pieces) {
                    if (Q[s_prev].count(candidate) && Q[s_prev][candidate] > max_q) {
                        max_q = Q[s_prev][candidate];
                        best_a = candidate;
                        found = true;
                    }
                }
                if (found) {
                    a_i = best_a;
                } else {
                    auto it = remaining_pieces.begin();
                    std::advance(it, rng() % remaining_pieces.size());
                    a_i = *it;
                }
            }
            
            // --- Take a_i, enter stage i, s_i = a_i ---
            State s_curr = a_i;
            current_sequence.push_back(a_i);
            
            // Calculate reward
            double r_i = 0.0;
            if (i == n_pieces - 1) {
                // Last piece placed, calculate full packing height
                double H = calculate_packing_height(current_sequence);
                r_i = (H > 0) ? (C / H) : 0.0;
                
                // Update Optimal
                if (H < min_height_opt) {
                    min_height_opt = H;
                    s_opt = current_sequence;
                }
            }
            
            // --- Update Q(s_{i-1}, a_i) ---
            // Max Q(s', a') for next state s_curr
            double max_q_next = 0.0;
            // Note: In this problem structure, next actions are from remaining pieces.
            // We must only look at valid next actions (not yet placed pieces).
            // Create temp set of next valid actions for lookahead
            std::unordered_set<PieceIndex> next_valid_actions = remaining_pieces;
            next_valid_actions.erase(a_i);
            
            if (!next_valid_actions.empty()) {
                double temp_max = -std::numeric_limits<double>::max();
                bool has_entry = false;
                for (PieceIndex next_a : next_valid_actions) {
                    if (Q[s_curr].count(next_a)) {
                         if (Q[s_curr][next_a] > temp_max) {
                             temp_max = Q[s_curr][next_a];
                             has_entry = true;
                         }
                    }
                }
                if (has_entry) max_q_next = temp_max;
            }
            
            double old_q = Q[s_prev][a_i];
            // Q(s, a) = Q(s, a) + alpha * [ r + gamma * max(Q(s', a')) - Q(s, a) ]
            Q[s_prev][a_i] = old_q + alpha * (r_i + gamma * max_q_next - old_q);
            
            // s <- s', move to next step
            s_prev = s_curr;
            remaining_pieces.erase(a_i);
        }
    }
    
    return s_opt;
}
```

**Algorithm 3:** Sarsa-learning for a 2D irregular packing problem
```cpp
// Algorithm 3: Sarsa-learning for 2D Irregular Packing
std::vector<PieceIndex> sarsa_learning_packing_optimization(int n_pieces, int m_episodes) {
    std::map<State, std::map<Action, double>> Q;
    
    std::vector<PieceIndex> s_opt;
    double min_height_opt = std::numeric_limits<double>::max();
    
    std::mt19937 rng(42);
    
    double alpha = 0.5;
    double gamma = 1.0;
    
    for (int t = 0; t < m_episodes; ++t) {
        std::vector<PieceIndex> current_sequence;
        std::unordered_set<PieceIndex> remaining_pieces;
        for(int i=0; i<n_pieces; ++i) remaining_pieces.insert(i);
        
        State s_prev = START_STATE;
        Action a_curr;
        
        // --- Initialize s0 and Choose a_i at s_{i-1} (first action) ---
        // Helper lambda for epsilon-greedy selection
        auto choose_action = [&](State s, const std::unordered_set<PieceIndex>& available) -> Action {
            std::uniform_real_distribution<> dist(0.0, 1.0);
            if (available.empty()) return -1;
            
            if (dist(rng) < EPSILON || Q[s].empty()) {
                auto it = available.begin();
                std::advance(it, rng() % available.size());
                return *it;
            } else {
                double max_q = -std::numeric_limits<double>::max();
                Action best_a = -1;
                bool found = false;
                for (PieceIndex cand : available) {
                    if (Q[s].count(cand) && Q[s][cand] > max_q) {
                        max_q = Q[s][cand];
                        best_a = cand;
                        found = true;
                    }
                }
                if (found) return best_a;
                else {
                    auto it = available.begin();
                    std::advance(it, rng() % available.size());
                    return *it;
                }
            }
        };
        
        a_curr = choose_action(s_prev, remaining_pieces);
        
        for (int i = 0; i < n_pieces; ++i) {
            // --- Take a_i, enter stage i, s_i = a_i ---
            State s_curr = a_curr; // s_i
            current_sequence.push_back(a_curr);
            remaining_pieces.erase(a_curr); // Remove used piece
            
            // Calculate reward
            double r_i = 0.0;
            if (i == n_pieces - 1) {
                double H = calculate_packing_height(current_sequence);
                r_i = (H > 0) ? (C / H) : 0.0;
                if (H < min_height_opt) {
                    min_height_opt = H;
                    s_opt = current_sequence;
                }
            }
            
            // --- Choose a'_{i} at s'_{i-1} (next action) ---
            Action a_next = -1;
            double q_next_val = 0.0;
            
            if (i < n_pieces - 1) {
                // Select next action from remaining pieces using policy
                a_next = choose_action(s_curr, remaining_pieces);
                if (Q[s_curr].count(a_next)) {
                    q_next_val = Q[s_curr][a_next];
                }
            }
            
            // --- Update Q(s_{i-1}, a_i) ---
            // Sarsa: Uses the Q-value of the actual next action chosen
            double old_q = Q[s_prev][a_curr];
            Q[s_prev][a_curr] = old_q + alpha * (r_i + gamma * q_next_val - old_q);
            
            // s <- s', a <- a'
            s_prev = s_curr;
            a_curr = a_next;
        }
    }
    
    return s_opt;
}
```

### Bibliography

- Hybrid Learning: Fang, J., Rao, Y., Zhao, X., Du, B., Fang, J., Rao, Y., Zhao, X., & Du, B. (2023). A Hybrid Reinforcement Learning Algorithm for 2D Irregular Packing Problems. Mathematics, 11(2). https://doi.org/10.3390/math11020327
- Q Learning: Dayan, P.; Watkins, C.J.C.H. Q-learning. Mach. Learn. 1992, 8, 279–292.
- Sarsa Learning: Sprague, N.; Ballard, D.H. Multiple-Goal Reinforcement Learning with Modular Sarsa(0). In Proceedings of the 18th international joint conference on Artificial intelligence, Acapulco, Mexico, 9–15 August 2003.
