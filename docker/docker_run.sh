# sudo docker run --gpus all -it \
#     --name orz_container \
#     --ipc=host \
#     -v $(pwd):/workspace \
#     -w /workspace \
#     fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest /bin/bash

# sudo docker run --gpus all -it \
#     --name orz_container \
#     --ipc=host \
#     --privileged \
#     -v $(pwd):/workspace \
#     -w /workspace \
#     fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest /bin/bash



# Remove existing container if it exists
if sudo docker ps -a --format '{{.Names}}' | grep -q '^orz_container$'; then
    sudo docker rm -f orz_container
fi

sudo docker run --gpus all -it \
    --name orz_container \
    --ipc=host \
    --tmpfs /dev/mqueue \
    -v $(pwd):/workspace:rw \
    -w /workspace \
    fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest /bin/bash

cd /fs-computility/mllm1/limo/workspace/stepfun/Open-Reasoner-Zero
export HF_ENDPOINT=https://hf-mirror.com
DEBUG_MODE=True python -m playground.orz_7b_ppo
