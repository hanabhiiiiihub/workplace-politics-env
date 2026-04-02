---
title: Workplace Politics Env
emoji: 🏢
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Workplace Politics Environment

An OpenEnv-compatible simulation where an AI agent must navigate workplace politics to get a proposal approved.

## Tasks
- **Level 1 (Easy):** Build coalitions, talk to the right people in the right order
- **Level 2 (Medium):** Race against a sabotaging Rival working behind the scenes  
- **Level 3 (Hard):** Compete against another team for the same budget

## API
- `POST /reset` — start a new episode
- `POST /step` — take an action
- `GET /state` — get full environment state

## Actions
- `schedule_meeting`, `share_info`, `request_support`, `observe`, `build_alliance`, `make_proposal`