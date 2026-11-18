# Running KernelForge on Kaggle / Colab

Moved from the top-level README. For the full runbook, see
[handoff/KAGGLE_RUNBOOK.md](handoff/KAGGLE_RUNBOOK.md).

## GPU session (Kaggle T4)

Import [`notebooks/kaggle_gpu_session.ipynb`](../notebooks/kaggle_gpu_session.ipynb) into Kaggle
(Accelerator: GPU T4, Internet: on). Or run directly:

```bash
# In a Kaggle notebook cell:
!git clone --depth 1 -b main https://github.com/Arsalan-05/KernelForge.git
%cd KernelForge
!pip install -q -r requirements-demo.txt
!python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"

# GPU-verify dataset + baseline eval (no fine-tuning):
!python scripts/gpu_session.py --budget-hours 11 --out /kaggle/working/kernelforge_outputs.zip

# Full run (+ QLoRA fine-tune + fine-tuned eval):
!python scripts/gpu_session.py --finetune --budget-hours 11 --out /kaggle/working/kernelforge_outputs.zip
```

All results are written to `kernelforge_outputs.zip`. See
[handoff/KAGGLE_RUNBOOK.md](handoff/KAGGLE_RUNBOOK.md) for the full runbook,
expected output files, resume instructions, and a post-session checklist.

## Run the demo (Kaggle/Colab)

Triton requires Linux + NVIDIA GPU. In a Kaggle/Colab GPU notebook cell:

```bash
!git clone https://github.com/Arsalan-05/KernelForge.git
%cd KernelForge
!pip install -q -r requirements-demo.txt
!python app/launch.py --share
```

This prints a temporary public URL (`Public: https://….trycloudflare.com`). Open it, pick an
op, and hit **Generate**. The kernel streams token by token, then goes through parse and harness
verification. Set **Samples** (1–4) and **Repair** (off/1/2) to run a kernel search.

**Note:** the demo currently runs the base model (not fine-tuned). The fine-tuned adapter can be
loaded with `--adapter training/checkpoints/<run>/adapter` once training is done.
