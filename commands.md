pip install torch==1.13.0+rocm5.2 torchvision==0.14.0+rocm5.2 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/rocm5.2
export HSA_OVERRIDE_GFX_VERSION=10.3.0
pacman -S rocm-hip-sdk rocm-opencl-sdk