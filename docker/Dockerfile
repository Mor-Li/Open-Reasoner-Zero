FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV DEBIAN_FRONTEND="noninteractive" \
    MAX_JOBS=8

# install apt dependencies
RUN apt-get update \
    && apt-get install -y openssh-server iputils-ping net-tools iproute2 traceroute netcat \
    && apt clean

# install pytorch_memlab
RUN pip install py-spy pytorch_memlab

# 复制预下载的 PyTorch `.whl` 文件到容器
COPY pytorch_wheels /tmp/pytorch_wheels

# 先安装 PyTorch 2.5.1、torchvision 0.20.1、torchaudio 2.5.1
RUN pip install --no-cache-dir --no-index --find-links=/tmp/pytorch_wheels \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121

# 安装 flash-attn
RUN pip install -v /tmp/pytorch_wheels/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# Remove the wheels after installation
RUN rm -rf /tmp/pytorch_wheels

RUN apt-get update && apt-get install -y \
    build-essential \
    libopenexr-dev \
    libxi-dev \
    libglfw3-dev \
    libglew-dev \
    libomp-dev \
    libxinerama-dev \
    libxcursor-dev \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖 (pip)
RUN pip install loguru tqdm ninja tensorboard \
    sentencepiece fire tabulate easydict \
    transformers==4.48.1 awscli rpyc pythonping \
    # 我猜这里应该不是非得0.18.0吧 不然他和pytorch2.5.1不兼容 然后和vllm也就不兼容了啊
    # torchvision==0.18.0 \ 
    hydra-core accelerate \
    redis opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp prometheus-client \
    omegaconf black==22.8.0 mypy-extensions pathspec tensorboardX nvitop antlr4-python3-runtime==4.11.0 \
    ray==2.40.0 deepspeed==0.16.0 vllm==0.6.5 peft

RUN apt update && apt install --fix-broken && apt install -y default-jre-headless openjdk-8-jdk
RUN eval $(curl -s deploy.i.basemind.com/httpproxy) && pip install git+https://github.com/facebookresearch/hydra.git

# set nccl.conf and nccl_env
COPY nccl.conf /etc/nccl.conf

# set time zone to CST
RUN apt update && apt install -y tzdata  \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt clean

# fix deepspeed bug
COPY parameter_offload.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/parameter_offload.py
COPY partitioned_param_coordinator.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/partitioned_param_coordinator.py

# launch script
WORKDIR /workspace/

# (pid=8159) Warning: Your installation of OpenCV appears to be broken: module 'cv2.dnn' has no attribute 'DictValue'.Please follow the instructions at https://github.com/opencv/opencv-python/issues/884 to correct your environment. The import of cv2 has been skipped.
# (pid=8159) WARNING 03-06 20:26:26 cuda.py:32] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
# (LLMActor pid=8159) WARNING 03-06 20:26:31 config.py:2171] Casting torch.float16 to torch.bfloat16.