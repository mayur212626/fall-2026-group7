# Setup

Run these commands from the project folder in a Linux Bash terminal with Conda available.

The environment file pins Python 3.12.14, pip 26.2.1, NumPy 2.5.2, PyTorch 2.12.1, and Torchvision 0.27.1. The PyTorch packages use the CUDA 12.6 build.

Create the environment once:

```bash
conda env create -f environment.yml
conda activate synthaug-bench
```

If the environment already exists, activate it before running the checks below.

Check package dependencies and record the system details:

```bash
python -m pip check
python scripts/check_environment.py
```

The report is saved to `runs/environment.json` and excluded from Git. A null package version means its metadata was not found. The report records GPU details from `nvidia-smi`; GPU computation is checked separately below.

Run a small GPU calculation and compare its result and gradients with the expected values:

```bash
python - <<'PY'
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
assert torch.cuda.is_available(), "CUDA is not available"
print("GPU:", torch.cuda.get_device_name(0))

x = torch.ones(256, 256, device="cuda", requires_grad=True)
loss = (x @ x).mean()
loss.backward()
torch.testing.assert_close(loss.detach(), torch.tensor(256.0, device="cuda"))
torch.testing.assert_close(x.grad, torch.full_like(x, 2 / 256))
print("GPU calculation and gradient checks passed.")
PY
```

These checks passed on 19 September 2026 with an NVIDIA A10G and driver 595.84. The installed environment was checked after creating the Conda environment and installing PyTorch separately. A fresh installation from the updated environment file still needs verification.

The file pins the core packages. A complete dependency lock, dataset checks, and model training checks will be added as the implementation progresses.
