conda activate opencompass_lmdeploy_118

python /fs-computility/mllm1/limo/workspace/stepfun/Open-Reasoner-Zero/volc_tools.py \
    --task-cmd '. /fs-computility/mllm1/limo/workspace/stepfun/Open-Reasoner-Zero/docker/container_commands.sh' \
    --log-level DEBUG \
    --num-gpus 8 \
    --num-replicas 1 \
    --task-name "orz_7b_ppo" \
    --queue-name "mllm1" \
    --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest' \
    --yes
    
# 确实只需要改这个queue name就行了 不需要改什么config file yaml
# 但是确实需要这个volc_tools.py文件的修改版本为了去掉刚开始的source conda的命令

python /fs-computility/mllm1/limo/workspace/stepfun/Open-Reasoner-Zero/volc_tools.py \
    --task-cmd '. /fs-computility/mllm1/limo/workspace/stepfun/Open-Reasoner-Zero/docker/container_commands.sh' \
    --log-level DEBUG \
    --num-gpus 8 \
    --num-replicas 1 \
    --task-name "orz_7b_ppo" \
    --queue-name "llmeval_volc" \
    --image 'fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest' \
    --yes