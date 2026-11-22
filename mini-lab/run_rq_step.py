"""One optimizer step on 64 released Industrial embeddings using upstream RQVAE."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream/rq"))
from models.rqvae import RQVAE


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    source = ROOT / "Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy"
    embeddings = np.load(source, mmap_mode="r", allow_pickle=False)
    batch = torch.tensor(embeddings[:64], dtype=torch.float32)
    model = RQVAE(in_dim=batch.size(1), num_emb_list=[16, 16, 16], e_dim=16,
                  layers=[64], kmeans_init=False, sk_epsilons=[0, 0, 0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.encoder.mlp_layers[1].weight.detach().clone()
    decoded, quantization_loss, _ = model(batch, use_sk=False)
    loss, reconstruction = model.compute_loss(decoded, quantization_loss, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        delta = (model.encoder.mlp_layers[1].weight - before).abs().max().item()
        codes = model.get_indices(batch)
    if not torch.isfinite(loss) or not delta or codes.shape != (64, 3):
        raise ValueError("Expected a finite loss, a parameter change and three codes per item")
    result = {"torch": torch.__version__, "source_shape": list(embeddings.shape),
              "sample_rows": [0, 63], "codebook_sizes": [16, 16, 16], "seed": 42,
              "completed_optimizer_steps": 1, "reconstruction_mse_before_step": reconstruction.item(),
              "total_loss_before_step": loss.item(), "encoder_weight_max_abs_delta": delta,
              "codes_after_step_shape": list(codes.shape), "row0_after_step": codes[0].tolist(),
              "distinct_codes_for_64_rows": len(set(map(tuple, codes.tolist())))}
    report = ROOT / "evidence/rq-step.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
