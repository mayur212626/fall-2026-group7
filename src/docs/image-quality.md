# Generated-image quality

`src/component/image_quality.py` scores a folder of generated images against real CIFAR-100 training images with two measures:

- **FID**: the Fréchet distance between Gaussians fitted to the Inception features of the two sets. Lower is closer to the real images.
- **Improved Precision & Recall** (k = 3 nearest neighbours): precision is the share of generated images that fall inside the region covered by the real images, recall the share of real images covered by the generated ones. Precision measures realism, recall measures variety.

## Protocol

- The generated images must already be 32×32 PNGs, the resolution the classifiers train on. The tool refuses any other size.
- Features come from clean-fid's "clean" mode: each 32×32 image is resized to 299×299 with PIL bicubic and passed through the standard Inception network, whose 2048 pool features are used. The network weights (about 100 MB) are downloaded on the first run.
- The real reference is the 5,000 training images of the 50-per-class budget. Score 5,000 generated images, 50 per class, so both sets have the same size and classes.
- The tool also scores a second set of 5,000 real training images, 50 per class, outside the 50-per-class budget. This real-versus-real value is the best a generator can reach with this protocol, so generated scores are read against it.
- Only training images are used; validation and test images are never touched.

The Fréchet distance is computed with NumPy from the symmetric form of the matrix square root. clean-fid 0.1.35's own function calls a SciPy option that recent SciPy releases removed, so it is not used; a unit test checks the result against the textbook formula.

## Run

```bash
python -m src.component.image_quality --generated data/synthetic/sd15_prompt \
    --output reports/Latex_report/tables/quality_sd15_prompt.json
```

The JSON file holds the counts, FID, precision and recall for the generated set and the real-versus-real set, the settings, the clean-fid version, the date and the git commit. Precision and recall hold full distance matrices in memory, about 0.8 GB at 5,000 images per set.
