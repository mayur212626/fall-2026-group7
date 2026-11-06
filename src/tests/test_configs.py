"""Every classifier has a training recipe in both arms (JSON only, no torch)."""

import ast
import json
import unittest
from pathlib import Path

CONFIGS = Path("src/component/configs")


def model_names() -> list[str]:
    """Keys of ``MODELS`` in models.py, read without importing torch."""
    tree = ast.parse(Path("src/component/models.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MODELS" for t in node.targets):
            return [key.value for key in node.value.keys]
    raise AssertionError("MODELS not found in models.py")


class ConfigTest(unittest.TestCase):
    def test_every_model_has_a_recipe_in_both_arms(self) -> None:
        names = model_names()
        self.assertIn("convnext_tiny", names)
        for init in ("pretrained", "scratch"):
            config = json.loads((CONFIGS / f"train_{init}.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(config["models"]), sorted(names), init)
            for name in names:
                self.assertIn(config["models"][name]["optimizer"], ("adamw", "sgd"))


if __name__ == "__main__":
    unittest.main()
