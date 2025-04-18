#!/bin/bash
cd /fs-computility/mllm1/limo/workspace/stepfun/Open-Reasoner-Zero
export HF_ENDPOINT=https://hf-mirror.com
DEBUG_MODE=True python -m playground.orz_7b_ppo_needlebench_mix_math_instruct

# # 保持容器运行并进入交互模式
# exec /bin/bash 