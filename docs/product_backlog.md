# Product Backlog

PDF version: [Product Backlog](product_backlog.pdf)

Product Goal: build a multi-agent rescue simulation where agents explore a
damaged area, detect targets, communicate useful information, learn from the
environment, and improve rescue strategy over time.

The Product Backlog is ordered by product value and project dependencies. It is
used as a planning guide and may be refined during the project.

| Priority | Item | Reason / Value | Estimate |
|---:|---|---|---:|
| 1 | Set up development infrastructure | The team needs GitLab, coding language, development environment, and base project structure before implementation can start. | 18h |
| 2 | Integrate CI and automated testing | Tests should run automatically when new code is uploaded, so the project remains stable during development. | 6h |
| 3 | Create damaged-area grid environment | The simulator needs a grid world with blocked cells, walls, valid positions, obstacles, and reproducible random generation. | 9h |
| 4 | Add Target A and Target B spawn mechanics | Rescue scenarios need different target types instead of generic targets. | 6h |
| 5 | Implement agent movement model | Agents must move through the environment while avoiding walls and obstacles. | 16h |
| 6 | Add central sensor and basic communication | The agent must be able to ask the central sensor for observations, and the system needs basic information exchange for later coordination. | 18h |
| 7 | Add simple visual feedback and simulation loop | The team needs to run and demonstrate a basic scenario with text output and metrics. | 16h |
| 8 | Implement autonomous exploration baseline | Agents should start exploring without manual movement decisions. | 18h |
| 9 | Define state-space data structure and reward logic | The learning system needs a clear representation of states, actions, transitions, and rewards. | 15h |
| 10 | Implement single-agent rescue learning logic and evaluate | A single agent should learn or improve behavior before adding multiple agents, and the team should evaluate different scenarios. | 30h |
| 11 | Support multiple agent starting positions | The simulator must be able to place a defined set of agents at their starting positions. | 10h |
| 12 | Extend agent-sensor communication for multiple agents | Multiple agents need sensor observations of obstacles, targets, and other agents to support cooperative rescue behavior. | 12h |
| 13 | Add central multi-agent reinforcement learning architecture | A central server should organize and optimize exploration and rescue strategies. | 30h |
| 14 | Add agent-to-agent communication with radius limit | Agents should only communicate with nearby agents, as required for distributed coordination. | 15h |
| 15 | Add distributed multi-agent learning architecture | Agents should optimize behavior as a society without relying on a central optimization server. | 24h |
| 16 | Expand reward logic and strategy evaluation for multi-agent | The system needs rewards and metrics to compare different strategies. | 12h |
| 17 | Validate results with test scenarios | The team needs reliable evaluation and performance estimates across different scenarios. | 8h |
| 18 | Optimize precision/performance | Improve the quality of predictions or rescue decisions based on validation results. | 12h |
| 19 | Integrate uncertainty in movement and sensor readings | Real-world conditions require uncertain movement and sensor readings. | 12h |
| 20 | Build graphical simulator / visual interface | The official project asks for a graphical simulator showing the damaged area, agents, obstacles, targets, and rescue process. | 18h |
| 21 | Prepare final live demo and presentation | The product must be understandable and demonstrable for final delivery. | 18h |
