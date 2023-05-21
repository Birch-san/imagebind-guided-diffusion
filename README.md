# Embedding Compare

Gonna try guiding stable-diffusion on spherical distance loss from ImageBind embeddings.

## Setup

Clone repository (including ImageBind submodule):

```bash
git clone --recursive --depth 1 https://github.com/Birch-san/imagebind-guided-diffusion.git
cd imagebind-guided-diffusion
```

Create a Conda environment. I'm naming this after Python 3.11 and CUDA 12.1:

```bash
conda create -n p311-cu121 python=3.11
conda activate p311-cu121
```

## Install dependencies

**Ensure you have activated the conda environment you created above.**

(Optional) treat yourself to latest nightly of PyTorch (and a compatible torchvision), with support for Python 3.11 and CUDA 12.1:

```bash
pip install --upgrade --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu121
```

Install the rest of the dependencies:

```bash
pip install -r requirements.txt
```

## Run:

From root of repository:

```bash
PYTHONPATH="./src:./lib/ImageBind:$PYTHONPATH" python -m scripts.guidance_play
```