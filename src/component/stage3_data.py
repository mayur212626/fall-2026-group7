"""Stage 3 datasets: Oxford Flowers-102 and HAM10000, stored as arrays with a split manifest.

Stage 3 repeats the benchmark on two domains at opposite distances from what
Stable Diffusion draws well: flower photos (close) and dermoscopy images of skin
lesions (far). Both use the CIFAR-100 protocol, with these differences:

- Images are stored at 224x224 (short side resized to 256 with antialiased
  bicubic resampling, then a center crop), the classifier input size, in one
  ``images.npy`` (N, 224, 224, 3) uint8 with ``labels.npy`` beside it. The
  generated images go through the same ``prepare_image``.
- The split manifest also holds the test indices, since the test images are in
  the same array.
- Both test sets are imbalanced, so the checkpoint is chosen by balanced
  accuracy (mean recall over the classes; for Flowers-102 this is the usual
  mean per-class accuracy).

Flowers-102 keeps its official split: train and validation have 10 images per
class and the remaining 6,149 are the test set. Budgets of 2, 5 and 10 images
per class are drawn from the official training images, nested, with split
seed 42.

HAM10000 has no official split, and one lesion often has several photos, so the
split is by lesion: within each class the lesions are shuffled with split seed
42 and assigned to test until it holds 20% of the class's images, then to
validation until 10%, and the rest to training. Budgets of 5, 10, 20 and 50
images per class take the training images in an order that uses every lesion
once before any lesion twice, so small budgets hold as many lesions as possible.

    python -m src.component.stage3_data download --dataset ham10000      # Flowers-102 downloads itself
    python -m src.component.stage3_data prepare --dataset flowers102
    python -m src.component.stage3_data prepare --dataset ham10000

This module needs only NumPy and Pillow (and torchvision to download Flowers-102).
"""

import argparse
import csv
import hashlib
import json
import logging
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

SPLIT_SEED = 42
STORED_SIZE = 224
RESIZE_SHORT_SIDE = 256

# Flowers-102 class names in label order (label 0 is the dataset's class 1), as used by CLIP.
FLOWERS102_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", "colt's foot",
    "king protea", "spear thistle", "yellow iris", "globe flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink and yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus",
    "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "air plant", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily",
]
# HAM10000 diagnosis codes in label order (alphabetical) and the names used in prompts and tables.
HAM10000_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM10000_CLASSES = ["actinic keratosis", "basal cell carcinoma", "benign keratosis", "dermatofibroma", "melanoma",
                    "melanocytic nevus", "vascular lesion"]
HAM10000_DOI = "doi:10.7910/DVN/DBW86T"
DATAVERSE = "https://dataverse.harvard.edu"
HAM10000_FILES = ("HAM10000_images_part_1", "HAM10000_images_part_2", "HAM10000_metadata")

DATASETS = {
    "flowers102": {
        "display": "Flowers-102",
        "classes": FLOWERS102_CLASSES,
        "budgets": ["2", "5", "10"],
        "pilot_budget": "5",
        "source": "torchvision Flowers102, official train/val/test split",
        "sd_per_class": 50,  # 5:1 at the largest budget (10 per class)
        "templates": (
            "a photo of {article} {name}, a type of flower.",
            "a close-up photo of {article} {name}, a type of flower.",
            "a photo of {article} {name} flower in a garden.",
            "a bright photo of {article} {name}, a type of flower.",
            "a photo of {article} {name} flower in its natural setting.",
            "a good photo of {article} {name}, a type of flower.",
            "a cropped photo of {article} {name} flower.",
            "a photo of a blooming {name}, a type of flower.",
        ),
    },
    "ham10000": {
        "display": "HAM10000",
        "classes": HAM10000_CLASSES,
        "budgets": ["5", "10", "20", "50"],
        "pilot_budget": "20",
        "source": f"HAM10000 (Tschandl et al., 2018), Harvard Dataverse {HAM10000_DOI}, CC BY-NC 4.0",
        "sd_per_class": 250,  # 5:1 at the largest budget (50 per class)
        "templates": (
            "a dermoscopy image of {name}.",
            "a dermoscopic image of {article} {name} skin lesion.",
            "a close-up dermoscopy image of {name} on the skin.",
            "a dermatoscope photo of {article} {name}.",
            "a clinical dermoscopy image showing {name}.",
            "a magnified dermoscopy image of {article} {name} lesion.",
            "a dermoscopy photo of {name}, skin cancer screening.",
            "a high quality dermoscopy image of {name}.",
        ),
        "test_fraction": 0.2,
        "validation_fraction": 0.1,
    },
}
SELECTION_METRIC = "balanced_accuracy"


def labels_sha256(labels: Sequence[int]) -> str:
    """SHA-256 of the label sequence (the same fingerprint as ``cifar_splits``)."""
    return hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()


def prepare_image(image: Image.Image, size: int = STORED_SIZE, short_side: int = RESIZE_SHORT_SIDE) -> np.ndarray:
    """Resize the short side to ``short_side`` (antialiased bicubic), center crop ``size`` x ``size``, RGB uint8."""
    image = image.convert("RGB")
    width, height = image.size
    scale = short_side / min(width, height)
    new_w, new_h = max(size, round(width * scale)), max(size, round(height * scale))
    image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return np.asarray(image.crop((left, top, left + size, top + size)), dtype=np.uint8)


def nested_budgets(order_by_class: dict[int, list[int]], budgets: Sequence[str]) -> dict[str, list[int]]:
    """Budget ``k`` takes the first ``k`` indices of every class's order, so every budget holds the smaller ones.

    Raises:
        ValueError: If a class has fewer training images than the largest budget.
    """
    largest = max(int(b) for b in budgets)
    out: dict[str, list[int]] = {b: [] for b in budgets}
    for label, order in sorted(order_by_class.items()):
        if len(order) < largest:
            raise ValueError(f"class {label} has {len(order)} training images, the largest budget needs {largest}")
        for b in budgets:
            out[b].extend(order[: int(b)])
    return {b: sorted(indices) for b, indices in out.items()}


def flowers_manifest(split_of: Sequence[str], labels: Sequence[int], budgets: Sequence[str],
                     seed: int = SPLIT_SEED) -> dict:
    """Manifest of the official Flowers-102 split; ``split_of[i]`` is ``"train"``, ``"val"`` or ``"test"``.

    The training images of each class are shuffled once with ``seed`` and the budgets take the first
    ``k`` of them.
    """
    split_of, labels = np.asarray(split_of), np.asarray(labels)
    rng = np.random.default_rng(seed)
    order = {int(c): rng.permutation(np.flatnonzero((split_of == "train") & (labels == c))).tolist()
             for c in np.unique(labels)}
    return {
        "split_seed": seed,
        "random_generator": "numpy PCG64",
        "split_rule": "official Flowers-102 train/val/test; budgets nested within the official training images",
        "validation_indices": np.flatnonzero(split_of == "val").tolist(),
        "test_indices": np.flatnonzero(split_of == "test").tolist(),
        "train_indices": nested_budgets(order, budgets),
    }


def lesion_split(lesions: Sequence[str], labels: Sequence[int], test_fraction: float, validation_fraction: float,
                 budgets: Sequence[str], seed: int = SPLIT_SEED) -> dict:
    """Split images by lesion, stratified by class, and draw nested training budgets.

    Within each class (in label order) the lesions are shuffled with one generator seeded with
    ``seed``. Whole lesions go to test until it holds ``test_fraction`` of the class's images, then to
    validation until ``validation_fraction``, and the rest to training. The training order lists the
    first image of every lesion (lesions in shuffled order), then every second image, and so on.

    Raises:
        ValueError: If a lesion has images of two classes, or a class is too small for the budgets.
    """
    lesions, labels = np.asarray(lesions), np.asarray(labels)
    by_lesion: dict[str, set] = {}
    for lesion, label in zip(lesions.tolist(), labels.tolist(), strict=True):
        by_lesion.setdefault(lesion, set()).add(label)
    mixed = [x for x, found in by_lesion.items() if len(found) > 1]
    if mixed:
        raise ValueError(f"lesion {mixed[0]} has images of more than one class")
    rng = np.random.default_rng(seed)
    validation, test, order = [], [], {}
    for label in np.unique(labels):
        in_class = np.flatnonzero(labels == label)
        ids = sorted(set(lesions[in_class].tolist()))
        images_of = {lesion: [] for lesion in ids}
        for i in in_class:
            images_of[lesions[i]].append(int(i))
        shuffled = [ids[i] for i in rng.permutation(len(ids))]
        test_target = round(test_fraction * len(in_class))
        val_target = round(validation_fraction * len(in_class))
        train_lesions, in_test, in_validation = [], 0, 0
        for lesion in shuffled:
            if in_test < test_target:
                test.extend(images_of[lesion])
                in_test += len(images_of[lesion])
            elif in_validation < val_target:
                validation.extend(images_of[lesion])
                in_validation += len(images_of[lesion])
            else:
                train_lesions.append(lesion)
        rounds = max((len(images_of[x]) for x in train_lesions), default=0)
        order[int(label)] = [images_of[x][r] for r in range(rounds) for x in train_lesions if r < len(images_of[x])]
    return {
        "split_seed": seed,
        "random_generator": "numpy PCG64",
        "split_rule": (f"by lesion_id, stratified by class: test {test_fraction:.0%}, validation "
                       f"{validation_fraction:.0%} of each class's images, the rest training; budgets take one "
                       "image per lesion before any lesion's second image"),
        "validation_indices": sorted(validation),
        "test_indices": sorted(test),
        "train_pool_size": sum(len(o) for o in order.values()),
        "train_indices": nested_budgets(order, budgets),
    }


def write_arrays(out_dir: Path, images: np.ndarray, labels: Sequence[int]) -> None:
    """Save ``images.npy`` and ``labels.npy`` (written to temporary files first)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, array in (("images", images), ("labels", np.asarray(labels, dtype=np.int64))):
        tmp = out_dir / f"{name}.tmp.npy"
        np.save(tmp, array)
        tmp.replace(out_dir / f"{name}.npy")


def load_images(data_dir: Path) -> tuple[np.ndarray, list[int]]:
    """Return the stored images (memory-mapped, so data-loader workers share them) and labels."""
    data_dir = Path(data_dir)
    images = np.load(data_dir / "images.npy", mmap_mode="r")
    labels = np.load(data_dir / "labels.npy").tolist()
    return images, labels


def prepare_flowers(raw_root: Path) -> tuple[np.ndarray, list[int], list[str], list[str]]:
    """Read the official splits (train, then val, then test) with torchvision; download if missing."""
    from torchvision.datasets import Flowers102

    images, labels, split_of, files = [], [], [], []
    for split in ("train", "val", "test"):
        dataset = Flowers102(str(raw_root), split=split, download=True)
        for i in range(len(dataset)):
            image, label = dataset[i]
            images.append(prepare_image(image))
            labels.append(int(label))
            split_of.append(split)
        files += [Path(p).name for p in getattr(dataset, "_image_files", [""] * len(dataset))]
    return np.stack(images), labels, split_of, files


def read_ham_metadata(path: Path) -> list[dict]:
    """Rows of the HAM10000 metadata file (comma- or tab-separated), sorted by ``image_id``."""
    text = Path(path).read_text(encoding="utf-8-sig")
    dialect = "excel-tab" if "\t" in text.splitlines()[0] else "excel"
    rows = [{k: v.strip().strip('"') for k, v in r.items()} for r in csv.DictReader(text.splitlines(), dialect=dialect)]
    for column in ("lesion_id", "image_id", "dx"):
        if column not in rows[0]:
            raise ValueError(f"{path} has no {column} column")
    return sorted(rows, key=lambda r: r["image_id"])


def find_ham_files(raw_root: Path) -> tuple[Path, dict[str, Path]]:
    """Locate the metadata file and every ``ISIC_*.jpg`` below ``raw_root`` (unzipping archives once)."""
    raw_root = Path(raw_root)
    for archive in sorted(raw_root.glob("HAM10000_images_part_*.zip")):
        target = raw_root / archive.stem
        if not target.exists():
            with zipfile.ZipFile(archive) as z:
                z.extractall(target)
    metadata = sorted(p for p in raw_root.rglob("HAM10000_metadata*") if p.suffix in (".csv", ".tab", ""))
    if not metadata:
        raise FileNotFoundError(f"no HAM10000_metadata file in {raw_root}")
    jpgs = {p.stem: p for p in raw_root.rglob("ISIC_*.jpg")}
    return metadata[0], jpgs


def prepare_ham(raw_root: Path) -> tuple[np.ndarray, list[int], list[str], list[str]]:
    """Read every metadata row's image in ``image_id`` order; return images, labels, lesion and image IDs."""
    metadata, jpgs = find_ham_files(raw_root)
    rows = read_ham_metadata(metadata)
    missing = [r["image_id"] for r in rows if r["image_id"] not in jpgs]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images of the metadata are missing, e.g. {missing[0]}")
    images = np.stack([prepare_image(Image.open(jpgs[r["image_id"]])) for r in rows])
    labels = [HAM10000_CODES.index(r["dx"]) for r in rows]
    return images, labels, [r["lesion_id"] for r in rows], [r["image_id"] for r in rows]


def download_ham(raw_root: Path) -> list[Path]:
    """Download the HAM10000 images and metadata from Harvard Dataverse into ``raw_root``.

    Lists the dataset's files with the Dataverse API and fetches the two image archives and the
    metadata (tabular files in their original format). Files already present are skipped.
    """
    import urllib.request

    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    url = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={HAM10000_DOI}"
    with urllib.request.urlopen(url, timeout=60) as response:
        files = json.load(response)["data"]["latestVersion"]["files"]
    saved = []
    for entry in files:
        data_file = entry["dataFile"]
        name = data_file.get("originalFileName") or data_file["filename"]
        if not name.startswith(HAM10000_FILES):
            continue
        path = raw_root / name
        if not path.exists():
            logging.info("downloading %s (%.0f MB)", name, data_file.get("filesize", 0) / 2**20)
            tmp = path.with_name(path.name + ".part")
            urllib.request.urlretrieve(f"{DATAVERSE}/api/access/datafile/{data_file['id']}?format=original", tmp)
            tmp.replace(path)
        saved.append(path)
    if len(saved) < len(HAM10000_FILES):
        raise RuntimeError(f"found {[p.name for p in saved]} on Dataverse; expected {HAM10000_FILES}")
    return saved


def build(dataset: str, raw_root: Path, out_dir: Path, manifest_path: Path) -> dict:
    """Prepare one dataset: arrays in ``out_dir`` and the split manifest at ``manifest_path``."""
    spec = DATASETS[dataset]
    if dataset == "flowers102":
        images, labels, split_of, files = prepare_flowers(raw_root)
        split = flowers_manifest(split_of, labels, spec["budgets"])
        ids = {"image_files": files}
    else:
        images, labels, lesions, image_ids = prepare_ham(raw_root)
        split = lesion_split(lesions, labels, spec["test_fraction"], spec["validation_fraction"], spec["budgets"])
        ids = {"image_ids": image_ids, "lesion_ids": lesions}
    write_arrays(out_dir, images, labels)
    manifest = {
        "dataset": spec["display"],
        "source": spec["source"],
        "num_classes": len(spec["classes"]),
        "classes": spec["classes"],
        "stored_size": STORED_SIZE,
        "preprocessing": f"short side to {RESIZE_SHORT_SIDE} (PIL bicubic, antialiased), center crop {STORED_SIZE}",
        "images": len(labels),
        "numpy_version": np.__version__,
        "labels_sha256": labels_sha256(labels),
        "shots_per_class": [int(b) for b in spec["budgets"]],
        **split,
        **ids,
    }
    check_manifest(manifest, labels)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_json(manifest), encoding="utf-8")
    return manifest


def manifest_json(manifest: dict) -> str:
    """JSON with one top-level key per line and every value on that line, so long index lists stay compact."""
    lines = [f"  {json.dumps(key)}: {json.dumps(value)}" for key, value in manifest.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def check_manifest(manifest: dict, labels: Sequence[int]) -> None:
    """Validation, test and the largest budget are disjoint, budgets are nested and balanced, lesions not shared.

    Raises:
        ValueError: On the first violated rule.
    """
    labels = np.asarray(labels)
    if labels.min() < 0 or labels.max() >= manifest["num_classes"]:
        raise ValueError(f"labels must be in range(0, {manifest['num_classes']})")
    validation, test = set(manifest["validation_indices"]), set(manifest["test_indices"])
    budgets = sorted(manifest["train_indices"], key=int)
    largest = set(manifest["train_indices"][budgets[-1]])
    if validation & test or validation & largest or test & largest:
        raise ValueError("validation, test and training images overlap")
    for small, big in zip(budgets, budgets[1:]):
        if not set(manifest["train_indices"][small]) <= set(manifest["train_indices"][big]):
            raise ValueError(f"budget {small} is not inside budget {big}")
    num_classes = manifest["num_classes"]
    for b in budgets:
        counts = np.bincount(labels[manifest["train_indices"][b]], minlength=num_classes)
        if not (counts == int(b)).all():
            raise ValueError(f"budget {b} does not have {b} images in every class")
    if "lesion_ids" in manifest:
        lesions = np.asarray(manifest["lesion_ids"])
        parts = [set(lesions[sorted(s)].tolist()) for s in (validation, test, set(manifest["train_indices"][budgets[-1]]))]
        if parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]:
            raise ValueError("a lesion appears in more than one split")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["download", "prepare"])
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--data-root", type=Path, help="default: data/<dataset>")
    parser.add_argument("--manifest", type=Path, help="default: src/component/configs/splits/<dataset>.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data_root = args.data_root or Path("data") / args.dataset
    manifest_path = args.manifest or Path("src/component/configs/splits") / f"{args.dataset}.json"

    if args.action == "download":
        if args.dataset != "ham10000":
            parser.error("only ham10000 needs a separate download (Flowers-102 downloads during prepare)")
        for path in download_ham(data_root / "raw"):
            logging.info("have %s", path)
        return
    manifest = build(args.dataset, data_root / "raw", data_root, manifest_path)
    sizes = {b: len(v) for b, v in manifest["train_indices"].items()}
    logging.info("%s: %d images, validation %d, test %d, training budgets %s", manifest["dataset"],
                 manifest["images"], len(manifest["validation_indices"]), len(manifest["test_indices"]), sizes)
    logging.info("saved %s/images.npy, labels.npy and %s", data_root, manifest_path)


if __name__ == "__main__":
    main()
