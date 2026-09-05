from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from .steering_repr import _atomic_json, _sha256_file


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def deterministic_positions(token_count: int, maximum: int) -> torch.Tensor:
    """Select the same number of evenly spaced aligned tokens from each base."""
    if token_count < 1 or maximum < 1:
        raise ValueError("token_count and maximum must be positive")
    if token_count <= maximum:
        return torch.arange(token_count, dtype=torch.long)
    return torch.tensor(
        [index * (token_count - 1) // (maximum - 1) for index in range(maximum)],
        dtype=torch.long,
    )


def _load_sample(
    path: Path, sample_per_base: int, device: torch.device
) -> torch.Tensor:
    record = torch.load(path, map_location="cpu", weights_only=False)
    values = record["activations"]
    positions = deterministic_positions(values.shape[1], sample_per_base)
    return values[:, positions].to(device=device, dtype=torch.float32)


def _covariance_apply(
    shard_paths: list[Path],
    matrix: torch.Tensor,
    sample_per_base: int,
) -> torch.Tensor:
    result = torch.zeros_like(matrix)
    for path in shard_paths:
        values = _load_sample(path, sample_per_base, matrix.device)
        centered = values - values.mean(0, keepdim=True)
        q = centered.reshape(-1, centered.shape[-1])
        result.add_(q.T @ (q @ matrix), alpha=1.0 / (len(shard_paths) * len(q)))
        del values, centered, q
    return result


def _fit_tokenwise_basis(
    shard_paths: list[Path],
    rank: int,
    sample_per_base: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    device = torch.device("cuda")
    first = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    hidden_size = int(first["activations"].shape[-1])
    del first
    center = torch.zeros(hidden_size, device=device)
    covariance_trace = 0.0
    sampled_token_counts = []
    for path in shard_paths:
        values = _load_sample(path, sample_per_base, device)
        sampled_token_counts.append(int(values.shape[1]))
        center += values.mean((0, 1)) / len(shard_paths)
        centered = values - values.mean(0, keepdim=True)
        covariance_trace += float(
            centered.square().sum() / centered.shape[0] / centered.shape[1]
        )
        del values, centered
    covariance_trace /= len(shard_paths)

    width = min(hidden_size, rank + oversample)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    subspace = torch.randn(hidden_size, width, generator=generator).to(device)
    subspace = torch.linalg.qr(subspace, mode="reduced").Q
    iteration_norms = []
    for _ in range(power_iterations + 1):
        applied = _covariance_apply(shard_paths, subspace, sample_per_base)
        iteration_norms.append(float(applied.norm()))
        subspace = torch.linalg.qr(applied, mode="reduced").Q
    applied = _covariance_apply(shard_paths, subspace, sample_per_base)
    reduced = subspace.T @ applied
    reduced = (reduced + reduced.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(reduced)
    order = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[order]
    basis = subspace @ eigenvectors[:, order[:rank]]
    basis = torch.linalg.qr(basis, mode="reduced").Q
    rayleigh = torch.diag(
        basis.T @ _covariance_apply(shard_paths, basis, sample_per_base)
    )
    orthogonality_error = float(
        (basis.T @ basis - torch.eye(rank, device=device)).abs().max()
    )
    report = {
        "algorithm": "deterministic randomized block power eigensolver",
        "centering": "per matched token across five roles",
        "base_weighting": "equal bases via fixed deterministic token sample",
        "sample_per_base_max": sample_per_base,
        "sampled_tokens_per_role": sum(sampled_token_counts),
        "sampled_role_centered_vectors": 5 * sum(sampled_token_counts),
        "oversample": oversample,
        "power_iterations": power_iterations,
        "seed": seed,
        "covariance_trace": covariance_trace,
        "leading_eigenvalues": rayleigh.detach().cpu().tolist(),
        "leading_standard_deviations": rayleigh.clamp_min(0).sqrt().cpu().tolist(),
        "captured_trace_fraction": float(rayleigh.sum() / covariance_trace),
        "iteration_operator_norms": iteration_norms,
        "orthogonality_max_abs_error": orthogonality_error,
    }
    return basis.cpu(), center.cpu(), report


def build_continuous_geometry(
    repo: Path,
    rank: int = 4,
    sample_per_base: int = 256,
    oversample: int = 12,
    power_iterations: int = 2,
    seed: int = 20260906,
) -> Path:
    if rank < 1 or oversample < 1 or power_iterations < 0:
        raise ValueError("invalid continuous geometry parameters")
    repr_root = repo / "research_outputs/phase3_steering_v2/repr_pre_mlp"
    repr_metadata_path = repr_root / "metadata.json"
    repr_metadata = json.loads(repr_metadata_path.read_text())
    probe_state_path = (
        repo
        / "research_outputs/phase3_steering_v2/representation_analysis/steering_state.pt"
    )
    probe_state = torch.load(probe_state_path, map_location="cpu", weights_only=False)
    manifest_path = repo / repr_metadata["manifest_path"]
    rows = _read_jsonl(manifest_path)
    train_ids = [int(row["base_id"]) for row in rows if row["split"] == "repr_train"]
    if len(train_ids) != 175:
        raise ValueError(
            f"expected 175 representation-train bases, found {len(train_ids)}"
        )
    layers = list(probe_state["candidate_layers_by_dev_accuracy"])
    output = repo / "research_outputs/phase3_continuous_v2/geometry"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "geometry_state.pt"
    if state_path.exists():
        raise FileExistsError(f"{state_path} exists")
    state = {
        "schema_version": 1,
        "hook": repr_metadata["hook"],
        "upstream_name": repr_metadata["upstream_name"],
        "manifest_sha256": _sha256_file(manifest_path),
        "probe_state_sha256": _sha256_file(probe_state_path),
        "candidate_layers": layers,
        "rank": rank,
        "sample_per_base": sample_per_base,
        "basis": {},
        "representation_center": {},
    }
    reports = []
    started = time.perf_counter()
    for layer in layers:
        layer_started = time.perf_counter()
        shard_paths = [
            repr_root / f"shards/layer-{layer:02d}/base-{base_id:03d}.pt"
            for base_id in train_ids
        ]
        missing = [path for path in shard_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} layer-{layer} shards")
        basis, center, diagnostics = _fit_tokenwise_basis(
            shard_paths,
            rank,
            sample_per_base,
            oversample,
            power_iterations,
            seed + layer,
        )
        state["basis"][layer] = basis
        state["representation_center"][layer] = center
        reports.append(
            {
                "layer": layer,
                "elapsed_seconds": round(time.perf_counter() - layer_started, 3),
                **diagnostics,
            }
        )
        _atomic_json(
            output / "run_state.json",
            {
                "status": "running",
                "completed_layers": [item["layer"] for item in reports],
                "remaining_layers": [
                    item
                    for item in layers
                    if item not in {report["layer"] for report in reports}
                ],
            },
        )
        print(
            f"continuous geometry layer {layer}: "
            f"trace4={diagnostics['captured_trace_fraction']:.4f} "
            f"orth={diagnostics['orthogonality_max_abs_error']:.2e} "
            f"in {reports[-1]['elapsed_seconds']:.1f}s",
            flush=True,
        )
        torch.cuda.empty_cache()
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, state_path)
    _atomic_json(
        output / "geometry_report.json",
        {
            "schema_version": 1,
            "method": "tokenwise across-role-centered continuous role covariance",
            "selection": "layers are top three by representation-dev probe accuracy",
            "representation_bases": len(rows),
            "representation_train_bases": len(train_ids),
            "rank": rank,
            "state_sha256": _sha256_file(state_path),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "layers": reports,
        },
    )
    _atomic_json(
        output / "run_state.json",
        {
            "status": "complete",
            "completed_layers": layers,
            "remaining_layers": [],
        },
    )
    return output
