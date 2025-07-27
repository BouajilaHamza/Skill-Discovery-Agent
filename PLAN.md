# Skill Discovery Agent (DIAYN, MiniGrid, PyTorch Lightning)

## Notes
- Project: Educational Skill Discovery Agent using DIAYN (optionally RE3) in MiniGrid.
- Priorities: Simplicity, modularity, clarity, and extensibility.
- All models must be lightweight (MLP/tiny CNN, low-res obs).
- Use PyTorch Lightning and CPU-only training.
- Log skill behavior/diversity; include clean code and comments.
- Use Python 3.x, gymnasium (MiniGrid), PyTorch Lightning (unless user changes mind).
- Avoid heavy dependencies; keep code minimal and educational.
- Project structure uses `src/agents`, `src/envs`, `src/models`, `src/scripts`, etc. (see user tree above).
- Place DIAYN agent in `src/agents/diayn_agent.py`, environment wrappers in `src/envs/`, models in `src/models/`, training script in `src/scripts/train.py`, configs in `configs/`.
- Confirm assumptions with user if unclear.

## Task List
- [ ] Define environment wrapper for MiniGrid (low-res obs, skill-conditioning) in `src/envs/`.
- [ ] Design model skeletons: encoder, policy, discriminator (MLP/tiny CNN) in `src/models/`.
- [ ] Implement DIAYN agent and training loop in `src/agents/diayn_agent.py` and `src/scripts/train.py`.
- [ ] Add logging for skill behavior/diversity (trajectories, skill labels).
- [ ] Provide config file snippet (YAML) and CLI run example.
- [ ] Suggest and implement a simple evaluation/visualization (e.g., print/plot trajectories).
- [ ] Comment and explain all code sections.
- [ ] Confirm user assumptions/preferences before finalizing modules.

## Current Goal
Draft/adapt code skeletons and plan for project structure.