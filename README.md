# MT-WAM

MT-WAM is a world–action model for language-conditioned robot manipulation. It combines a Wan2.2 video backbone with an action transformer, using motion supervision from CoTracker and visual feature supervision from DINOv2.

This repository provides data preparation, training and evaluation code for LIBERO and RoboTwin, together with LIBERO-Plus evaluation.

The project is organized into [source code](src/mtwam/), [configurations](configs/), [scripts](scripts/), [benchmark adapters](experiments/) and [tests](tests/).

## Installation

Use Linux with Python 3.10, PyTorch 2.7.1, CUDA 12.8 and FFmpeg. System dependencies include Git, Git LFS and EGL for LIBERO; RoboTwin also uses a C++ compiler, CMake, Ninja, `unzip`, the CUDA toolkit and Vulkan.

Run the following from the repository root:

```bash
conda create -n mtwam -c conda-forge python=3.10 ffmpeg=6.1
conda activate mtwam
python -m pip install setuptools==80.9.0 wheel==0.45.1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[train]'
python scripts/setup_dependencies.py teachers
```

### LIBERO / LIBERO-Plus

Set up each benchmark in its own environment. Set `benchmark` to `libero` or `libero-plus`:

```bash
benchmark=libero
python -m pip install -e '.[libero]'
python scripts/setup_dependencies.py "$benchmark"
export LIBERO_CONFIG_PATH="$PWD/.deps/${benchmark}-config"
export MUJOCO_GL=egl
```

### RoboTwin

```bash
python -m pip install -e '.[robotwin]'
python scripts/setup_dependencies.py robotwin
python scripts/download_robotwin_assets.py
```

## Models and data

Download the models and datasets to these default locations:

| Resource | Location |
| --- | --- |
| [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e): video DiT, VAE and text encoder | `pretrained_models/Wan2.2-TI2V-5B/` |
| [Wan tokenizer](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/37ec512624d61f7aa208f7ea8140a131f93afc9a/google/umt5-xxl) | `pretrained_models/Wan2.1-T2V-1.3B/google/umt5-xxl/` |
| [CoTracker3](https://huggingface.co/facebook/cotracker3/tree/bf55ea50d4390e1820a267f131cd6587240fb2c5) | `pretrained_models/scaled_offline.pth` |
| [DINOv2 ViT-B/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) | `pretrained_models/dinov2_vitb14_pretrain.pth` |
| [LIBERO data](https://huggingface.co/datasets/yuanty/LIBERO-fastwam/tree/ee018b997c430bb12b5bf3c892d744798c5a2f91) | `data/libero_mujoco3.3.2/` |
| [RoboTwin data](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam/tree/aac262c35d02cc71b2f6ef670bd65fd9f2bb2547) | `data/robotwin2.0/` |

Extract LIBERO's four `*_no_noops_lerobot.tar.gz` archives into its data directory. Merge and extract RoboTwin's `robotwin2.0.tar.gz.part-*` archives into `data/robotwin2.0/`, and place its `dataset_stats.json` there.

Paths are configured in [configs/_train_common.yaml](configs/_train_common.yaml); dataset settings are in [configs/data/](configs/data/).

## Training

Select `libero` or `robotwin`, then initialize the action backbone, prepare text embeddings and launch training:

```bash
benchmark=libero
python scripts/preprocess_action_dit_backbone.py --config-name=train_${benchmark} --output=pretrained_models/action_dit_${benchmark}.pt
python scripts/precompute_text_embeds.py --config-name=train_${benchmark}
bash scripts/train_zero1.sh 8 --config-name=train_${benchmark}
```

Training settings are in [train_libero.yaml](configs/train_libero.yaml) and [train_robotwin.yaml](configs/train_robotwin.yaml). Runs are saved under `runs/`; use `output_dir=...` to choose a location.

## Evaluation

Place your trained checkpoint, `config.yaml` and `dataset_stats.json` in `checkpoint/`. Run the corresponding command in the benchmark environment.

**LIBERO**

```bash
python experiments/libero/run_libero_manager.py --config-name=eval_libero ckpt=./checkpoint/step.pt
```

**LIBERO-Plus**

```bash
python experiments/libero/run_libero_manager.py --config-name=eval_libero_plus ckpt=./checkpoint/step.pt
```

**RoboTwin C2R**

```bash
python experiments/robotwin/run_robotwin_manager.py --config-name=eval_robotwin ckpt=./checkpoint/step.pt
```

Evaluation settings are in [configs/](configs/).

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -m 'not gpu'
python scripts/check_release.py
```

## License and acknowledgments

MT-WAM contributions use the [MIT license](LICENSE). Component licenses are listed in [NOTICE](NOTICE) and [licenses/](licenses/). CoTracker uses [CC BY-NC 4.0](licenses/CoTracker-CC-BY-NC-4.0.txt), and CuRobo uses [NVIDIA's noncommercial research and evaluation terms](licenses/CuRobo-NVIDIA.txt).

We thank the authors of Wan, DiffSynth-Studio, CoTracker, DINOv2, LeRobot, LIBERO, LIBERO-Plus, RoboTwin, robosuite and PyTorch3D.
