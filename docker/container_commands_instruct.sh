#!/bin/bash
cd /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero
DEBUG_MODE=True python -m playground.orz_7b_ppo_needlebench_mix_math_instruct

# # 保持容器运行并进入交互模式
# exec /bin/bash 