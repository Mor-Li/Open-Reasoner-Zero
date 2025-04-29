#!/bin/bash
cd /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero
export HF_ENDPOINT=https://hf-mirror.com
DEBUG_MODE=True python -m playground.orz_7b_ppo_atc_prm_32k