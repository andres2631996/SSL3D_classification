import torch


def restore_best_score(callback, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    callback_state = ckpt["callbacks"].get(type(callback).__qualname__)
    if callback_state and "best_model_score" in callback_state:
        callback.best_model_score = callback_state["best_model_score"]
        callback.best_model_path = callback_state.get("best_model_path", ckpt_path)
