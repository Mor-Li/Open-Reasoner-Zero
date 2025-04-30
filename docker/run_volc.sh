# 确实只需要改这个queue name就行了 不需要改什么config file yaml
# 但是确实需要这个volc_tools.py文件的修改版本为了去掉刚开始的source conda的命令

# 1 node for default sequence length
python /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/volc_tools.py \
    --task-cmd 'cd /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero && DEBUG_MODE=True python -m playground.orz_7b_ppo_atc_prm' \
    --log-level DEBUG \
    --num-gpus 8 \
    --num-replicas 1 \
    --task-name "orz_7b_ppo_atc_prm" \
    --queue-name "llmeval_volc" \
    --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest' \
    --yes

# 1 node for default sequence length to llmeval_volc queue and train from instruct model and needle 128
python /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/volc_tools.py \
    --task-cmd 'cd /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero && DEBUG_MODE=True python -m playground.orz_7b_ppo_atc_prm_instruct_128' \
    --log-level DEBUG \
    --num-gpus 8 \
    --num-replicas 1 \
    --task-name "orz_7b_ppo_atc_prm_instruct_128" \
    --queue-name "llmeval_volc" \
    --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest' \
    --yes

# 1 node for default sequence length and vanilla ppo reward
python /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/volc_tools.py \
    --task-cmd 'cd /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero && DEBUG_MODE=True python -m playground.orz_7b_ppo_atc_vanillarm' \
    --log-level DEBUG \
    --num-gpus 8 \
    --num-replicas 1 \
    --task-name "orz_7b_ppo_atc_vanillarm" \
    --queue-name "llmeval_volc" \
    --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest' \
    --yes


# 1 node for default sequence length and vanilla ppo reward to mllm1 queue
python /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/volc_tools.py \
    --task-cmd 'cd /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero && DEBUG_MODE=True python -m playground.orz_7b_ppo_atc_vanillarm' \
    --log-level DEBUG \
    --num-gpus 8 \
    --num-replicas 1 \
    --task-name "orz_7b_ppo_atc_vanillarm" \
    --queue-name "mllm1" \
    --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest' \
    --yes


# 2 nodes to llmeval_volc queue
# multi node use ray[default] so orz:v2
python /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/ray_deploy.py \
  --task-cmd "cd /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero && ./docker/ray_2nodes.sh" \
  --n-nodes 2 \
  --n-gpus-per-node 8 \
  --queue-name "llmeval_volc" \
  --task-name "orz_7b_ppo_atc_prm_instruct_128" \
  --log-level DEBUG \
  --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:v2' \
  --yes
