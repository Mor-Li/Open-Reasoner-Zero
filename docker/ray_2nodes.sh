# 方式一：Worker0 上启动 Head 节点，其余 Worker 作为普通节点
cd /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero 
if [[ $MLP_ROLE_INDEX == 0 ]]; then
    ray start --head
    sleep 20
    # 启动你的训练脚本（替换为实际脚本名）
    python -m playground.orz_7b_ppo_atc_prm > ${MLP_LOG_PATH}/ray_head_start.log 2>&1
else
    ray start --address=${MLP_WORKER_0_HOST}:6379 > ${MLP_LOG_PATH}/ray_worker_start.log 2>&1
    tail -f /dev/null  # 或者其他保持脚本运行的命令
fi
