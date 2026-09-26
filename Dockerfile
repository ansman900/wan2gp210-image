# Matches setup_wan2gp_5090.sh except models and the GPU-only torch.cuda check.
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CONDA_DIR=/opt/conda \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/opt/conda/envs/wan2gp210/bin:/opt/conda/bin:/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        build-essential ninja-build pkg-config \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN if [ ! -e /usr/local/cuda-13.0 ]; then ln -s /usr/local/cuda /usr/local/cuda-13.0; fi \
    && nvcc --version \
    && test -f /usr/local/cuda/include/cusparse.h \
    && test -f /usr/local/cuda/include/cublas_v2.h

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda config --set always_yes yes \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true \
    && conda create -y -n wan2gp210 python=3.11.14 \
    && conda clean -afy

SHELL ["/bin/bash", "-lc"]

RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate wan2gp210 \
    && python -m pip install -U pip "setuptools<=75.8.2" ninja packaging

RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate wan2gp210 \
    && pip install --no-cache-dir \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu130

WORKDIR /opt/Wan2GP

RUN git clone --depth 1 https://github.com/deepbeepmeep/Wan2GP.git /opt/Wan2GP

RUN python - <<'PY'
from pathlib import Path
p = Path("/opt/Wan2GP/wgp.py")
text = p.read_text()
needle = "torch.set_float32_matmul_precision('high')"
if needle in text:
    print("wgp.py already patched")
else:
    lines = text.splitlines(keepends=True)
    out, done = [], False
    for line in lines:
        out.append(line)
        if not done and line.strip() in ("import torch", "import torch;"):
            out.append("torch.set_float32_matmul_precision('high')\n")
            done = True
    if not done:
        raise SystemExit("could not find first import torch in wgp.py")
    p.write_text("".join(out))
    print("patched wgp.py")
PY

RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate wan2gp210 \
    && pip install --retries 20 --timeout 120 --no-cache-dir -r /opt/Wan2GP/requirements.txt \
    && pip install --no-cache-dir hf_transfer

RUN <<'EOS'
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate wan2gp210
pip uninstall -y sageattention || true
git clone --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/SageAttention
cd /tmp/SageAttention
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=1
python -c "
import os, sys, runpy
import torch.utils.cpp_extension as ext
ext._check_cuda_version = lambda *a, **k: None
os.chdir('/tmp/SageAttention')
os.environ['CUDA_HOME'] = '/usr/local/cuda-13.0'
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
os.environ['MAX_JOBS'] = '1'
sys.argv = ['setup.py', 'install']
runpy.run_path('setup.py', run_name='__main__')
"
python -c "import sageattention; print('sage ok', getattr(sageattention, '__version__', 'ok'))"
rm -rf /tmp/SageAttention
EOS

ENV WAN2GP_SITE=/opt/conda/envs/wan2gp210/lib/python3.11/site-packages
ENV LD_LIBRARY_PATH=/opt/conda/envs/wan2gp210/lib/python3.11/site-packages/nvidia/cu13/lib:/opt/conda/envs/wan2gp210/lib/python3.11/site-packages/torch/lib

RUN echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate wan2gp210' >> /root/.bashrc \
    && echo 'unset CUDA_HOME' >> /root/.bashrc

WORKDIR /workspace
EXPOSE 7860
CMD ["bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && conda activate wan2gp210 && unset CUDA_HOME && cd /opt/Wan2GP && python wgp.py --listen --share"]
