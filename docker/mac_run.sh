# Remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q '^orz_container$'; then
    docker rm -f orz_container
fi


docker run --platform=linux/amd64 -it \
    --name orz_container \
    --ipc=host \
    --tmpfs /dev/mqueue \
    -v $(pwd):/workspace:rw \
    -w /workspace \
    open-reasoner-zero:latest /bin/bash


DEBUG_MODE=True python3 -m playground.orz_14m_ppo_mini