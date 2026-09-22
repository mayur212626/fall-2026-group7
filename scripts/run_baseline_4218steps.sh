set -eu
cd "$HOME/synthaug-bench"

for seed in 0 1 2 3 4; do
    for model in resnet50 convnext_tiny vit_b_16 swin_t; do
        printf '\nStarting %s, seed %s\n' "$model" "$seed"
        python -u -m scripts.train_classifier \
            --model "$model" \
            --augmentation real_only \
            --full-data \
            --seed "$seed" \
            --config configs/cifar100_baseline_4218steps.json \
            --output "runs/baseline_4218steps/${model}_seed${seed}"
    done
done

echo "All 20 baseline runs completed."
