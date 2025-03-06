pip install --no-cache-dir --no-index --find-links=/home/limo/Desktop/Open-Reasoner-Zero/docker/pytorch_wheels \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

python -c "import torch; print(f'PyTorch ABI compatibility (CXX11 ABI): {getattr(torch._C, \"_GLIBCXX_USE_CXX11_ABI\", \"Unknown\")}')"
# (test_torch) limo@holmes-XPS-8940:~/Desktop/Open-Reasoner-Zero/docker/pytorch_wheels$ python -c "import torch; print(f'PyTorch ABI compatibility (CXX11 ABI): {getattr(torch._C, \"_GLIBCXX_USE_CXX11_ABI\", \"Unknown\")}')"
# PyTorch ABI compatibility (CXX11 ABI): False

