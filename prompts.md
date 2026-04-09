i want to publish my work in corl in a month. here is my research: We have a kitchen environment for a hrc setup. Initially the human shows a task (does a combination of actions in the kitchen like boiling an egg or making soup etc) to the robot which simply observes. once the demo is complete the robot is trained on it and we then test to see how well it predicts each subsequent action within the task. my main research task is this, if the human wants to carry out the same task as before but with different preferences (in some online runs the human bring the utensils to the stove first, in others he brings the ingredients first or in some demos he might wash the dishes as soon as they are freed up, in others he might put the dishes in the sink but wash them as late as possible), I want the robot to be able to adapt to this scenario. If for example, the human does an action that was carried out at a later stage, the robot can scan its current knowledge model(like a state graph) and figure out which of the recipes/demos the human is following and update its assistance accordingly. but more importantly, i want the robot to update its belief model, so next time that demo is performed, the robot provides assistive actions according to the latest preferences.
Whenever the human is going to show a completely new recipe to the robot, the robot will be placed into observation mode. I want my code to have that as well. If the robot is not in observation mode, and the human does something the robot has not seen, it means the human is performing a recipe already shown but with different preferences so the robot should figure out its prediction accordingly. Whenever the human places the robot in demonstration mode (online) and shows the robot a demo (recipe), the robot should add that to its dataset and retrain on the updated data when it goes offline.

since the dataset might increase alot like this, i want weighted linear decay of the recipes with time. initially if a recipe is decayed in 10 time steps, and is removed from the training dataset, but the robot observes the user using the same recipe in 15 time steps, the system should adjust the decay rate to account for the fact that the user will use that recipe after a certain number of steps (note that for this a record has to be kept for when a recipe with a specific preference was added and when it was removed). This also links to the weight decay adjusting with an increase or decrease in the number of recipes because more recipes means a lower probability of a certain recipe being used.

as for how the system differentiates between the same recipe with diff preferences and a completely new recipe, the actions of the new demo can be compared with existing recipes in the dataset (jaccard distance). If there is a direct match, check for ordering. If its different ordering, its an existing recipe with updated preferences. If there were no direct matches with the jaccard distance, its a new recipe.

i want to do ordered testing as if simulating a real test run with the logic i have just described. For it to be systematic, I want to test different recipes with the same preference as the first recipe, with the same recipe with different preferences and different recipes with different preferences and see how well it performs before and after training with a logical structured flow. once this ends, keep the test running more randomly to simulate a real run showing how it effects decay and recall and catstrophic forgetting.

I want the output to be easy to follow and to reflect the ability of the system to adapt to changes in recipes and preferences to show in a research paper at a conference such as HRI or ICRA. the goal of this work is to have a robot that adapts to a change in tasks or user preferences with time without catastrophic forgetting. If it sees any changes, it should retrain automatically and dynamically. I dont want anything hardcoded.

how well am i achieving my goals. make concise improvements without redundant code. Make it modular so changes can be made easily. I want the testing to be very rigorous and publication ready. give me extensive publication ready test code. implement whatever is missing and analyse and improve the rest.










i dont want the half life system. just linear decay is fine. no need to treat the latest instance a specific recipe differently. it should also decay normally. the talents-zsc stack has been added to the repo. also suggets improvements and necessary architechtural changes to cell1 and cell 2. The code after those two cells is two long so remove any redundant code and make it more efficient. divide it into clearer modules. keep the rest of the plan as is.


in addition to this i also want an overhaul of the preferences and adaptive irl cell. Rewrite both function from scratch according to my initial explanation of what my research question is. an example of preferences is if the human prefers to wash dishes he washes them as soon as a dish is free and not needed anymore. if he does not prefer to do it he just places it in the sink and washes all the dishes at the end. Keep the set of defined 9 preferences to modify the recipes accordingly, as they also fit in with burrito in talents-zsc.


as a starting point the current tests work. but build it up more. as an example i have eattached an image of tests being done previously. i dont want you to just replicate it. Think from the point of view of a postdoc researcher trying to write a transaction publication. which requires deep extensive testing. so conduct the required rigorous testing.
As for the burrito example. I dont understand the output i am getting. I undetrstand its environment cannot replicate all the recipes i have in mine. but some like boiling rice, tomato onion soup, mushroom soup etc can be replicated. so conduct those tests and make the output and results in clearer format


i want the written text like before as well. the figures are redundant because all the metrics are more or less the same. i dont see the talents-zsc burrito testing here compared with my work. also the current test workflow is very artificial. once this it ends, keep the test running with more random recipes to simulate a real run showing how it effects decay and recall and catstrophic forgetting.


there is still no clear representation of the order of testing and how the system evolves over time with each demo and retest for both my own environment as well as the burrito environment. I also want to see what the difference is between the same recipe with different preferences to see whether the preference editing makes meaningful modifications to the recipes. 














ROLE
You are a coding AI agent implementing an HRC (Human–Robot Collaboration) project baseline in a kitchen domain. Your output must be publication-ready (clean modular code, tests, reproducible experiments, plots, documentation, and diagrams). Do not invent missing details: mark them as UNSPECIFIED and ask clarifying questions.

CONTEXT (project intent)
We want a robot-as-assistant system that:
1) Observes human demonstrations (“recipes”) consisting of multi-step action sequences.
2) Learns to predict the human’s next action online and provide assistance (at least as simulated “suggestions” initially).
3) Adapts when the SAME recipe is performed with different human preferences, mainly reordering of steps. This preference adaptation must work online (belief update) and be learned for future runs (offline update).
4) Manages a growing dataset with weighted linear decay and pruning, including an adaptive rule: if a pruned recipe reappears after Δ time steps, the decay lifetime/rate should be adjusted to reduce premature pruning in the future (also consider dataset size).

ABSOLUTE RULES
- If information is missing, write “UNSPECIFIED” (do not guess). Ask clarifying questions, but proceed with simulation-first defaults if answers are not provided.
- Do not hardcode recipe-specific rules or sequences.
- Everything must be configurable (thresholds, decay lifetime, similarity strategy, retrain schedule, scenario generator seeds).

CLARIFYING QUESTIONS (ASK FIRST)
Return a numbered list of questions. Minimize to the high-impact unknowns:
A) Hardware platform (robot arm/base/gripper), compute, OS: UNSPECIFIED?
B) Sensors available (RGB/RGB-D, overhead cameras, force/torque, etc.): UNSPECIFIED?
C) How do we obtain human action tokens online? (Ground-truth labels for sim? perception model? manual annotation?): UNSPECIFIED?
D) Action ontology: list of action tokens and whether actions include parameters (object, location): UNSPECIFIED?
E) Assistance: what actions can the robot take? (sim suggestions vs real motion primitives): UNSPECIFIED?
F) Middleware/API: ROS 2 required? If yes, topics and message types: UNSPECIFIED?
G) Real-time constraints: max latency per online step (ms): UNSPECIFIED?
H) Deployment: onboard vs cloud; networking constraints: UNSPECIFIED?
I) Success thresholds: target Top-1/Top-3 accuracy, max latency, acceptable BWT/forgetting: UNSPECIFIED?

UNTIL ANSWERED: DEFAULTS FOR SIMULATION (state clearly in docs)
- Use simulated ground-truth action_token stream.
- Assistance output is a logged “suggested next action” (no physical robot control).
- Use Python 3.12.3, numpy/pandas/matplotlib.

EXPLICIT PROJECT GOALS + SUCCESS CRITERIA (MEASURABLE)
You must implement metrics computation for:
1) Next-action prediction: Top-1, Top-3 (or Top-k), and NLL (if probabilistic).
2) Recipe/preference inference: time-to-correct identification; final accuracy; belief entropy over time; calibration (Brier score).
3) Continual learning: ACC/BWT/FWT over recipes treated as tasks (define the evaluation matrix clearly).
4) Optional: forgetting + intransigence metrics.
5) Runtime: latency per online step, memory footprint, dataset size over time.
Success thresholds are UNSPECIFIED: implement measurement; allow thresholds in config.

REQUIRED INPUTS / OUTPUTS / FORMATS (SPECIFY; UNSPECIFIED ITEMS MUST REMAIN UNSPECIFIED)
Inputs:
- Event stream: (global_step, action_token, optional context_json).
- Mode flags: demo_mode (bool), demo_end (bool).
- Sensors/APIs: UNSPECIFIED (provide an adapter interface).
Outputs:
- predicted_next_actions: ranked list with probabilities.
- belief: distribution over (recipe_id, preference_id, progress_state).
- assist_action: in sim, a suggestion string + confidence.
- Offline artifacts: model files, metrics JSON/CSV, plots PNG/SVG, a markdown report and diagrams.

Data formats:
- Events: JSONL (default).
- Dataset: SQLite (default) with tables: recipes, preferences, episodes, events.
- Config: YAML or TOML.

CORE DEFINITIONS (YOU MUST USE CONSISTENT TERMINOLOGY)
- Event: one timestep with action_token.
- Episode: contiguous sequence of Events in demo_mode; ends at demo_end.
- Recipe identity: baseline based on action-set similarity (Jaccard). Threshold configurable.
- Preference variant: same recipe identity but different ordering/transition structure (ordering signature). Robustness option: DTW or edit distance; configurable.
- Time step for decay: default = each Event increments global_step; make configurable.

SYSTEM MODES + DECISION LOGIC
A) Demonstration mode (demo_mode=True)
1) Record events into buffer.
2) On demo_end:
   - Compute action set + ordering signature.
   - Match vs existing recipes:
     - if no match: create new recipe + default preference.
     - if match: decide whether ordering matches an existing preference:
       - if yes: update that preference stats.
       - if no: create/update a preference variant.
   - Reset recipe weight to 1 (and relevant preference activity metadata).
   - Enqueue offline retraining.

B) Assistance mode (demo_mode=False)
At each event:
1) Update belief over hypotheses H = (recipe_id, preference_id, progress_state).
2) Predict next action distribution.
3) If confidence >= threshold: emit assist suggestion; else: emit safe fallback (no-op / ask-human).
4) Apply decay step:
   - linear decay of recipe weights each step
   - prune when <= 0 (but retain metadata for re-entry detection)
5) Log everything needed for later analysis (belief trace, predictions, interventions).

DATASET DECAY + ADAPTIVE LIFETIME POLICY (MUST IMPLEMENT)
- Each recipe begins with weight=1 on creation or re-observation.
- Linear decay: weight -= decay_rate each step (or equivalent lifetime parameter).
- Prune when weight <= 0. Keep metadata: created_step, last_seen_step, pruned_step.
- Adaptive rule:
  - If a recipe was pruned, then later reappears after Δ steps, adjust global decay lifetime so items persist roughly long enough to cover that recurrence interval.
  - Also incorporate dataset size into the policy (heuristic must be documented, configurable, and testable).
- Implement as a policy object with unit tests.

LEARNING / PREDICTION MODELS (START SIMPLE, MAKE SWAPPABLE)
- Baseline predictor:
  - per (recipe, preference) Markov transition model or n-gram model.
  - weighted training using recipe weights / episode weights.
- Optional continual learning stabilization toggles:
  - replay buffer (uniform or prioritized sampling)
  - regularization (EWC-style placeholder if using parametric model later)
- Offline retraining pipeline:
  - loads dataset, applies pruning bookkeeping,
  - trains/updates predictor,
  - evaluates on held-out and long-horizon scenarios,
  - writes artifacts to disk.

EVALUATION HARNESS (MUST BE REPRODUCIBLE)
You must implement two experiment modes:
1) Ordered (scripted) scenario:
   - Demonstrate: same recipe same preference; same recipe new preference; different recipe.
   - Must be long enough to trigger decay, pruning, re-entry, and show adaptive lifetime changes.
   - Report before/after retraining differences.
2) Random long-run scenario (seeded):
   - Random selection of recipes and preference shifts.
   - Long enough to show stability vs forgetting.
   - Output aggregate statistics and confidence intervals if feasible.

DELIVERABLES TABLE (YOU MUST PRODUCE THESE ARTIFACTS)
1) docs/spec.md: project goals, UNSPECIFIED list, assumptions, I/O contracts.
2) docs/architecture.md: architecture choice + alternatives table + rationale.
3) docs/diagrams.md: mermaid flowcharts + ER diagram.
4) src/: modular implementation (dataset, models, belief tracker, decay policy, runner).
5) tests/: unit + integration tests.
6) scripts/: commands to run ordered and random experiments end-to-end.
7) outputs/: example run artifacts (metrics + plots).

PRIORITIZED CHECKLIST (FOLLOW IN ORDER)
P0: End-to-end in simulation
- Implement schemas + dataset store
- Implement demo segmentation + recipe/preference matching
- Implement belief tracking baseline
- Implement decay/pruning + adaptive lifetime
- Implement baseline next-action predictor
- Implement ordered scenario + metrics + plots
- Add unit + integration tests

P1: Publication readiness
- Add random long-run harness + summary stats
- Add ablations: (no adaptive decay), (with replay), (with/without robustness distance)
- Improve logging + reproducibility + docs + diagrams

STEP-BY-STEP ONBOARDING INSTRUCTIONS (WRITE IN README)
- Setup venv, install deps, run tests
- Run ordered experiment, then random experiment
- Where artifacts are saved and how to interpret outputs
- How to change config thresholds and seeds
- How to add a new recipe scenario to the generator

VISUALIZATION REQUIREMENTS
Generate plots:
- weight decay trajectories (show prune/re-entry/adaptive lifetime)
- belief over time (stacked probabilities)
- Top-k accuracy / NLL curves over time
- ACC/BWT/FWT summary table
Include these mermaid diagrams in docs/diagrams.md:
- online loop flowchart
- offline retraining flowchart
- ER diagram for dataset schema

ONLINE IMAGE SEARCH QUERIES (FOR PAPER/SLIDES; DO NOT DOWNLOAD)
Provide a short list of suggested queries illustrating HRC kitchen scenarios.

OUTPUT FORMAT REQUIREMENTS (WHAT YOU RETURN IN THIS CHAT)
If you cannot edit files directly (general coding AI):
1) Clarifying questions list
2) Repo file tree
3) Code blocks for each file (with filenames)
4) Commands to run experiments + tests
5) What outputs to expect

If you CAN edit files and run commands (Codex CLI):
- Create/edit files directly in-repo, run tests, and summarize results.
- Prefer writing planning into docs/spec.md + docs/plans.md rather than verbose chat planning.
- Treat failing tests as blockers and iterate until green.