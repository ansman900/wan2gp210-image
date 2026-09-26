# Wan2GP 5090 image: CUDA 13.0 toolkit + PyTorch 2.10 cu130 + SageAttention 2
# Build on any machine with Docker (GPU not required if using prebuilt wheels).
#
#   docker build -t YOURUSER/wan2gp210:cu130 .
#   docker push YOURUSER/wan2gp210:cu130

FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/envs/wan2gp210/bin:/opt/conda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs wget curl ca-certificates \
        build-essential ninja-build pkg-config \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda config --set always_yes yes \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Exact env from your Vast script
RUN conda create -y -n wan2gp210 python=3.11.14 \
    && conda clean -afy

SHELL ["/bin/bash", "-lc"]

RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate wan2gp210 \
    && pip install --no-cache-dir \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu130

WORKDIR /opt/Wan2GP
RUN git clone --depth 1 https://github.com/deepbeepmeep/Wan2GP.git /opt/Wan2GP

# Same matmul patch as your install script
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
    && pip install --no-cache-dir -r /opt/Wan2GP/requirements.txt \
    && pip install --no-cache-dir hf_transfer

# SageAttention 2 for torch 2.10 + CUDA 13.
# Prefer a prebuilt wheel if you have a URL; otherwise compile against the CUDA 13 toolkit in this image.
# If compile fails on your builder, comment this block and install a known-good .whl instead.
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate wan2gp210 \
    && pip install --no-cache-dir sageattention==2.2.0 --no-build-isolation \
    || pip install --no-cache-dir sageattention --no-build-isolation

# Convenience: drop into the right env in interactive shells
RUN echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate wan2gp210' >> /root/.bashrc \
    && echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate wan2gp210' >> /etc/bash.bashrc

WORKDIR /workspace
EXPOSE 7860

# Vast Jupyter/SSH launch modes replace ENTRYPOINT. Keep a simple default anyway.
CMD ["bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && conda activate wan2gp210 && cd /opt/Wan2GP && python wgp.py --listen --share"]
