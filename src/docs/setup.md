# Setup

Run these commands from the project folder in a Linux Bash terminal with Conda available.

The environment file pins Python 3.12.14, pip 26.2.1, NumPy 2.5.2, PyTorch 2.12.1, Torchvision 0.27.1 and Matplotlib 3.11.2. The PyTorch packages use the CUDA 12.6 build.

Create the environment once:

```bash
conda env create -f environment.yml
conda activate synthaug-bench
```

If the environment already exists, activate it and add any package that was pinned later, for example `pip install matplotlib==3.11.2`.

Check that the installed packages are consistent:

```bash
python -m pip check
```

Every training run records the Python, PyTorch, Torchvision and NumPy versions, the GPU, the weight version and the git commit in its own `config.json`, so the environment of each result stays traceable.

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

Download CIFAR-100 to `data/cifar100/` (about 170 MB; `data/` is excluded from Git):

```bash
python -c "from torchvision.datasets import CIFAR100; CIFAR100('data/cifar100', train=True, download=True); CIFAR100('data/cifar100', train=False, download=True)"
```

The split manifest in `src/component/configs/splits/cifar100.json` stores a fingerprint of the training labels, and the training runner refuses to start if the downloaded labels do not match it. [Data splits](data-splits.md) explains the manifest.

Run the unit tests:

```bash
python -m unittest discover -s src/tests
```

Between 25 and 27 September 2026 the installed environment on the NVIDIA A10G (driver 595.91.07) reported PyTorch 2.12.1+cu126, Torchvision 0.27.1+cu126 and CUDA available, and the unit tests passed. A fresh installation from the current environment file has not been verified yet, and the dependencies are pinned at the top level only; a complete lock file is still missing.
