# Sprint 2 Backlog

Sprint: Sprint 2  
Dates: May 6 - May 27  
Team capacity: 4 people x 18h = 72h

## Sprint Goal

Create the first working damaged-area simulator with grid environment, target
spawning, agent movement, central sensor communication, and simple visual
output.

## Selected Product Backlog Items

| Priority | Item | Estimate |
|---:|---|---:|
| 1 | Finish grid environment: blocked cells, walls, valid movements, obstacles not overlapping starting points, random generation using seeds, tests. | 9h |
| 2 | Fix Target A / Target B spawning instead of generic targets. | 6h |
| 3 | Add central sensor model and basic communication with agent: what and how sensor tells agent and vice versa, what sensor knows, tests. | 18h |
| 4 | Add simple visual output and connect all parts: text grid, config loading, simulation loop, metrics, integration test. | 16h |
| 5 | Make movement model for agents: allowed movements, avoid obstacles, tests, central sensor compatibility. | 16h |

## Sprint Backlog And Task Assignment

| Person | Tasks | Hours |
|---|---|---:|
| Adriana | Finish grid environment: blocked cells, walls, valid movements, obstacles not overlapping starting points, random generation using seeds, tests. Fix Target A / Target B spawning instead of generic targets. | 15h |
| Cristina | Add central sensor model and basic communication with agent: what and how sensor tells agent and vice versa, what sensor knows, tests. | 18h |
| xxx | Add simple visual output and connect all parts: text grid, config loading, simulation loop, metrics, integration test. | 16h |
| xxx | Make movement model for agents: allowed movements, avoid obstacles, tests, central sensor compatibility. | 16h |
| Everyone | Code review, CI/test fixes, merge fixes, meetings. | 7h |

## Capacity

Planned work: 65h  
Buffer: 7h for code review, meetings, general fixes, and communication.

## Definition Of Done

A Sprint 2 item is done when:

- the implementation works with the current project structure
- tests are added or updated
- the feature can be demonstrated or verified
- the merge request is reviewed by another teammate
- the work does not break existing CI checks
