# 确实只需要改这个queue name就行了 不需要改什么config file yaml
# 但是确实需要这个volc_tools.py文件的修改版本为了去掉刚开始的source conda的命令

# 1 node for default sequence length
python /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/volc_tools.py \
    --task-cmd '. /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/docker/container_commands.sh' \
    --log-level DEBUG \
    --num-gpus 8 \
    --num-replicas 1 \
    --task-name "orz_7b_ppo_atc_prm" \
    --queue-name "llmeval_volc" \
    --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest' \
    --yes

# 1 node for 32k to llmeval_volc queue
python /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/volc_tools.py \
    --task-cmd '. /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/docker/container_commands_32k.sh' \
    --log-level DEBUG \
    --num-gpus 8 \
    --num-replicas 1 \
    --task-name "orz_7b_ppo_atc_prm_32k" \
    --queue-name "llmeval_volc" \
    --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest' \
    --yes

# 1 node for 32k to mllm1 queue
python /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/volc_tools.py \
    --task-cmd '. /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/docker/container_commands_32k.sh' \
    --log-level DEBUG \
    --num-gpus 8 \
    --num-replicas 1 \
    --task-name "orz_7b_ppo_atc_prm_32k" \
    --queue-name "mllm1" \
    --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest' \
    --yes


# 2 nodes to llmeval_volc queue
python /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/ray_deploy.py \
  --task-cmd ". /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/docker/container_commands_32k_2node.sh" \
  --n-nodes 2 \
  --n-gpus-per-node 8 \
  --queue-name "q-20241107085952-dnfrk" \
  --task-name "orz_7b_ppo_atc_prm_32k_2nodes" \
  --log-level DEBUG \
  --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest' \
  --yes

# 2 nodes to mllm1 queue

python /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/ray_deploy.py \
  --task-cmd ". /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero/docker/container_commands_32k_2node.sh" \
  --n-nodes 2 \
  --n-gpus-per-node 8 \
  --queue-name "q-20241107090119-5rpvq" \
  --task-name "orz_7b_ppo_atc_prm_32k_2nodes" \
  --log-level DEBUG \
  --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest' \
  --yes