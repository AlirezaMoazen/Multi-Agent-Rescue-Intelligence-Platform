# Sprint 3 Backlog

PDF version: [Sprint 3 Backlog](sprint_3.pdf)

Sprint: Sprint 3  
Dates: May 27 - June 10  
Team capacity: 4 people x 18h = 72h

## Sprint Goal

Build the first single-agent learning prototype: the agent should explore the
rescue grid autonomously, use a clear state/action/reward representation, learn
from repeated episodes, and produce basic evaluation results across different
scenarios.

## Selected Product Backlog Items

| Priority | Item | Estimate |
|---:|---|---:|
| 1 | Implement autonomous exploration baseline: agents should start exploring without manual movement decisions. | 18h |
| 2 | Define state-space data structure and reward logic: the learning system needs a clear representation of states, actions, transitions, and rewards. | 15h |
| 3 | Implement single-agent rescue learning logic and evaluate, part 1: create the first training loop so a single agent can learn or improve behavior before adding multiple agents. | 15h |
| 4 | Implement single-agent rescue learning logic and evaluate, part 2: evaluate the trained single agent across scenarios and compare it against the baseline. | 15h |

## Sprint Backlog And Task Assignment

| Person | Tasks | Hours |
|---|---|---:|
| Cristina | Implement autonomous exploration baseline: autonomous move selection, visited-cell tracking, valid movement use, deterministic baseline behavior, and tests. | 18h |
| Alireza | Define state-space data structure and reward logic: state representation, action format, reward rules, episode termination rules, and tests/documentation for the learning contract. | 15h |
| Mustafa | Implement single-agent learning logic, part 1: Q-table structure, action selection, Q-value update, repeated training episodes, and basic training metrics. | 15h |
| Adriana | Implement single-agent learning logic, part 2: evaluation runner, scenario comparison, baseline comparison, success-rate/steps/reward metrics, and report-ready output. | 15h |
| Everyone | Code review, CI/test fixes, merge fixes, integration support, and sprint meetings. | 9h |


## Capacity

Planned work: 63h  
Buffer: 9h for code review, meetings, integration fixes, CI/test fixes, and
coordination between learning, movement, sensor, runner, and visualization.

## Definition Of Done

A Sprint 3 item is done when:

- the implementation works with the current grid, movement, runner, and shared structures
- tests are added or updated
- the feature can be demonstrated or verified with at least one scenario
- metrics or output are clear enough for comparison
- the merge request is reviewed by another teammate
- the work does not break existing CI checks
