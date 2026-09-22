import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.checkpoint import load_checkpoint, save_checkpoint


def update(model, optimizer, step):
    optimizer.param_groups[0]["lr"] = 0.01 * (1 - step / 10)
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(torch.randn(8, 4)), torch.randint(0, 2, (8,))).backward()
    optimizer.step()


class CheckpointTests(unittest.TestCase):
    def test_resumed_updates_match_uninterrupted_training(self):
        for optimizer_class, options in ((torch.optim.SGD, {"momentum": 0.9}), (torch.optim.AdamW, {})):
            with self.subTest(optimizer=optimizer_class.__name__), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(11)
                model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 2))
                optimizer = optimizer_class(model.parameters(), lr=0.01, **options)
                for step in range(3):
                    update(model, optimizer, step)
                path = Path(directory) / "last.pt"
                config = {"seed": 11, "steps": 6}
                save_checkpoint(path, model, optimizer, {"run_config": config, "step": 3})
                for step in range(3, 6):
                    update(model, optimizer, step)
                expected_model = copy.deepcopy(model.state_dict())
                expected_optimizer = copy.deepcopy(optimizer.state_dict())
                expected_rng = torch.get_rng_state()
                expected_cuda = torch.rand(4, device="cuda") if torch.cuda.is_available() else None

                restored = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 2))
                restored_optimizer = optimizer_class(restored.parameters(), lr=1.0, **options)
                state = load_checkpoint(path, restored, restored_optimizer, config)
                self.assertEqual(state["step"], 3)
                for step in range(state["step"], 6):
                    update(restored, restored_optimizer, step)
                torch.testing.assert_close(restored.state_dict(), expected_model, rtol=0, atol=0)
                torch.testing.assert_close(restored_optimizer.state_dict(), expected_optimizer, rtol=0, atol=0)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                if expected_cuda is not None:
                    torch.testing.assert_close(torch.rand(4, device="cuda"), expected_cuda, rtol=0, atol=0)

    def test_configuration_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            path = Path(directory) / "last.pt"
            save_checkpoint(path, model, optimizer, {"run_config": {"seed": 0}})
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, optimizer, {"seed": 1})

    def test_failed_save_keeps_the_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            path = Path(directory) / "last.pt"
            config = {"seed": 0}
            save_checkpoint(path, model, optimizer, {"run_config": config, "step": 1})
            original = path.read_bytes()
            with patch("scripts.checkpoint.torch.save", side_effect=OSError("Disk write failed")):
                with self.assertRaises(OSError):
                    save_checkpoint(path, model, optimizer, {"run_config": config, "step": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(load_checkpoint(path, model, optimizer, config)["step"], 1)


if __name__ == "__main__":
    unittest.main()
