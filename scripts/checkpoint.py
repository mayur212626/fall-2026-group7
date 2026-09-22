import torch


def save_checkpoint(path, model, optimizer, state):
    checkpoint = {
        **state,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, run_config):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["run_config"] != run_config:
        raise ValueError("The checkpoint configuration differs from this run.")
    model.load_state_dict(state.pop("model"))
    optimizer.load_state_dict(state.pop("optimizer"))
    torch.set_rng_state(state.pop("torch_rng"))
    cuda_rng = state.pop("cuda_rng")
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    return state
