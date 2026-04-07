---
title: Workplace Politics Env
emoji: 🏢
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Workplace Politics — OpenEnv Environment

A real-world AI training environment simulating organizational change management.
The agent must navigate workplace politics to get a proposal approved.

## Characters

| Character | Role | Agenda |
|---|---|---|
| Boss | Final approver | Hates surprises |
| Rival | Peer manager | Works against you |
| Friend | Ally | Folds under pressure |
| Gatekeeper | Admin | Controls Boss access |
| Fence-Sitter | Neutral | Sides with winner |

## Tasks

| Task | Difficulty | Max Steps |
|---|---|---|
| level_1_easy | Easy | 15 |
| level_2_medium | Medium | 12 |
| level_3_hard | Hard | 10 |

## Baseline Scores (llama-3.1-8b-instant via Groq)

| Level | Score | Outcome |
|---|---|---|
| Easy | 0.4590 | partial |
| Medium | 0.4560 | partial |
| Hard | 0.4380 | partial |
| Average | 0.4510 | — |

## API Endpoints

- `POST /reset` — start episode
- `POST /step` — take action
- `GET /state` — get full state
- `GET /grade` — get episode score

## Run Baseline
```bash
export HF_TOKEN=your_groq_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant
export ENV_URL=https://hanabhi-workplace-politics-env.hf.space
python inference.py
```