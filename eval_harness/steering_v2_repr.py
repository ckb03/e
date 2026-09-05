from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Config
from .steering_data import ROLES
from .steering_repr import _atomic_json, _git_commit, _load_local_model, _sha256_file

ROLE_TO_INDEX = {role: index for index, role in enumerate(ROLES)}
DEFAULT_LAYERS = tuple(range(0, 24, 2))


class _CaptureFinished(Exception):
    pass


class PreMlpCapture:
    """Capture the exact GPT-OSS `all_pre_mlp_hidden_states` representation."""

    def __init__(self, layers, selected_layers: Iterable[int]) -> None:
        self.selected_layers = tuple(selected_layers)
        self.last_layer = max(self.selected_layers)
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.handles = [
            layers[layer].mlp.register_forward_pre_hook(self._make_hook(layer))
            for layer in self.selected_layers
        ]

    def _make_hook(self, layer: int):
        def hook(_module, inputs):
            if self.positions is None:
                raise RuntimeError("pre-MLP capture positions were not configured")
            hidden = inputs[0]
            positions = self.positions.to(hidden.device)
            batches = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            self.values[layer] = (
                hidden[batches, positions]
                .detach()
                .to(device="cpu", dtype=torch.bfloat16)
            )
            if layer == self.last_layer:
                raise _CaptureFinished

        return hook

    def reset(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def collect_v2_representation_activations(
    repo: Path,
    config_path: Path,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    resume: bool = False,
) -> Path:
    manifest_path = repo / "eval_data/steering_repr_v2_manifest.jsonl"
    rows = _read_jsonl(manifest_path)
    if len(rows) != 250:
        raise ValueError(f"expected 250 v2 bases, found {len(rows)}")
    config = Config.load(config_path)
    output_dir = repo / "research_outputs/phase3_steering_v2/repr_pre_mlp"
    shard_root = output_dir / "shards"
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "schema_version": 1,
        "manifest_path": str(manifest_path.relative_to(repo)),
        "manifest_sha256": _sha256_file(manifest_path),
        "model_id": config.model_id,
        "model_config_fingerprint": config.fingerprint(),
        "hook": "GPT-OSS layer.post_attention_layernorm output / MLP input",
        "upstream_name": "all_pre_mlp_hidden_states",
        "selected_layers": list(layers),
        "roles": list(ROLES),
        "token_sampling": "all aligned content tokens, maximum 1024 per base",
        "sequence_weighting": "equal underlying base in downstream fits",
        "git_commit": _git_commit(repo),
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        invariant = (
            "manifest_sha256",
            "model_id",
            "model_config_fingerprint",
            "hook",
            "selected_layers",
            "roles",
            "token_sampling",
        )
        mismatch = [key for key in invariant if previous.get(key) != metadata[key]]
        if mismatch:
            raise ValueError(f"v2 activation resume mismatch: {mismatch}")
        if not resume:
            raise FileExistsError(f"{output_dir} exists; pass --resume")
        metadata = previous
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for layer in layers:
            (shard_root / f"layer-{layer:02d}").mkdir(parents=True, exist_ok=True)
        _atomic_json(metadata_path, metadata)
    completed = {
        base_id
        for base_id in range(250)
        if all(
            (shard_root / f"layer-{layer:02d}/base-{base_id:03d}.pt").exists()
            for layer in layers
        )
    }
    pending = [row for row in rows if int(row["base_id"]) not in completed]
    _atomic_json(
        output_dir / "run_state.json",
        {
            "status": "running" if pending else "complete",
            "completed_base_ids": sorted(completed),
            "remaining_base_ids": [int(row["base_id"]) for row in pending],
        },
    )
    if not pending:
        return output_dir
    tokenizer, model = _load_local_model(config)
    if max(layers) >= len(model.model.layers):
        raise ValueError("selected pre-MLP layer does not exist")
    metadata.update(
        {
            "num_model_layers": len(model.model.layers),
            "hidden_size": model.config.hidden_size,
            "model_type": type(model).__name__,
        }
    )
    _atomic_json(metadata_path, metadata)
    capture = PreMlpCapture(model.model.layers, layers)
    started = time.perf_counter()
    try:
        for row in pending:
            base_id = int(row["base_id"])
            prompts = [row["role_variants"][role]["prompt"] for role in ROLES]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            sequence_length = encoded.input_ids.shape[1]
            token_count = int(row["content_token_count"])
            positions = []
            for role_index, role in enumerate(ROLES):
                variant = row["role_variants"][role]
                prompt_length = int(encoded.attention_mask[role_index].sum())
                left_padding = sequence_length - prompt_length
                role_positions = list(
                    range(
                        left_padding + int(variant["content_token_start"]),
                        left_padding + int(variant["content_token_end"]),
                    )
                )
                observed = encoded.input_ids[role_index, role_positions].tolist()
                if observed != row["content_token_ids"]:
                    raise AssertionError(
                        f"runtime alignment failed for base {base_id}, role {role}"
                    )
                positions.append(role_positions)
            position_tensor = torch.tensor(positions, dtype=torch.long)
            capture.reset(position_tensor)
            model_inputs = {
                key: value.to(config.device) for key, value in encoded.items()
            }
            case_started = time.perf_counter()
            try:
                with torch.inference_mode():
                    model.model(**model_inputs, use_cache=False)
            except _CaptureFinished:
                pass
            if set(capture.values) != set(layers):
                raise RuntimeError(
                    f"base {base_id} missing pre-MLP layers: "
                    f"{sorted(set(layers) - set(capture.values))}"
                )
            for layer in layers:
                payload = {
                    "schema_version": 1,
                    "base_id": base_id,
                    "split": row["split"],
                    "layer": layer,
                    "roles": list(ROLES),
                    "content_sha256": row["content_sha256"],
                    "content_token_count": token_count,
                    "activations": capture.values[layer],
                }
                target = shard_root / f"layer-{layer:02d}/base-{base_id:03d}.pt"
                temporary = target.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, target)
            completed.add(base_id)
            remaining = [
                int(item["base_id"])
                for item in rows
                if int(item["base_id"]) not in completed
            ]
            _atomic_json(
                output_dir / "run_state.json",
                {
                    "status": "running" if remaining else "complete",
                    "completed_base_ids": sorted(completed),
                    "remaining_base_ids": remaining,
                },
            )
            print(
                f"v2 base {base_id}: {token_count} tokens x 5 roles x "
                f"{len(layers)} pre-MLP layers in "
                f"{time.perf_counter() - case_started:.3f}s",
                flush=True,
            )
    finally:
        capture.close()
    metadata["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    metadata["status"] = "complete"
    metadata["completed_bases"] = len(completed)
    _atomic_json(metadata_path, metadata)
    _atomic_json(
        output_dir / "run_state.json",
        {
            "status": "complete",
            "completed_base_ids": sorted(completed),
            "remaining_base_ids": [],
        },
    )
    return output_dir


def _load_layer_records(root: Path, layer: int) -> list[dict]:
    paths = sorted((root / f"shards/layer-{layer:02d}").glob("base-*.pt"))
    if len(paths) != 250:
        raise ValueError(f"layer {layer} has {len(paths)} of 250 shards")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def _split_features(records: list[dict], split: str):
    selected = [record for record in records if record["split"] == split]
    features = []
    labels = []
    weights = []
    for record in selected:
        values = record["activations"]
        token_count = values.shape[1]
        features.append(values.reshape(-1, values.shape[-1]))
        labels.append(
            torch.arange(len(ROLES), dtype=torch.long).repeat_interleave(token_count)
        )
        weights.append(torch.full((len(ROLES) * token_count,), 1.0 / token_count))
    x = torch.cat(features)
    y = torch.cat(labels)
    sample_weight = torch.cat(weights)
    sample_weight *= len(sample_weight) / sample_weight.sum()
    return x, y, sample_weight, selected


def _weighted_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    sample_weight: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    chunk_size: int = 32768,
) -> dict:
    device = torch.device("cuda")
    confusion = torch.zeros((len(ROLES), len(ROLES)), dtype=torch.float64)
    correct_weight = 0.0
    total_weight = 0.0
    loss_weight = 0.0
    role_correct = torch.zeros(len(ROLES), dtype=torch.float64)
    role_total = torch.zeros(len(ROLES), dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, len(x), chunk_size):
            end = min(len(x), start + chunk_size)
            x_batch = x[start:end].float().to(device)
            y_batch = y[start:end].to(device)
            sw = sample_weight[start:end].to(device)
            logits = x_batch @ weight.T + bias
            prediction = logits.argmax(-1)
            losses = F.cross_entropy(logits, y_batch, reduction="none")
            correct = prediction.eq(y_batch)
            correct_weight += float((correct * sw).sum())
            loss_weight += float((losses * sw).sum())
            total_weight += float(sw.sum())
            for target in range(len(ROLES)):
                target_mask = y_batch.eq(target)
                role_total[target] += float(sw[target_mask].sum())
                role_correct[target] += float(
                    (correct[target_mask] * sw[target_mask]).sum()
                )
                for predicted in range(len(ROLES)):
                    confusion[target, predicted] += float(
                        sw[target_mask & prediction.eq(predicted)].sum()
                    )
    row_normalized = confusion / confusion.sum(1, keepdim=True)
    return {
        "base_balanced_accuracy": correct_weight / total_weight,
        "base_balanced_cross_entropy": loss_weight / total_weight,
        "per_role_accuracy": {
            role: float(role_correct[index] / role_total[index])
            for index, role in enumerate(ROLES)
        },
        "confusion_matrix_base_balanced_rows": row_normalized.tolist(),
        "token_examples": len(x),
        "underlying_bases": len({record["base_id"] for record in []}),
    }


def _fit_upstream_style_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    sample_weight: torch.Tensor,
    c_value: float = 5e-3,
    max_iter: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit multinomial L2 logistic regression with a full-batch L-BFGS solver."""
    device = torch.device("cuda")
    x_device = x.float().to(device)
    y_device = y.to(device)
    sw_device = sample_weight.to(device)
    weight = torch.zeros((len(ROLES), x.shape[-1]), device=device, requires_grad=True)
    bias = torch.zeros(len(ROLES), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_iter * 2,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    calls = 0
    final_parts = {}

    def closure():
        nonlocal calls, final_parts
        optimizer.zero_grad(set_to_none=True)
        logits = x_device @ weight.T + bias
        losses = F.cross_entropy(logits, y_device, reduction="none")
        data_loss = (losses * sw_device).sum() / sw_device.sum()
        penalty = weight.square().sum() / (2 * c_value * sw_device.sum())
        loss = data_loss + penalty
        loss.backward()
        calls += 1
        final_parts = {
            "objective": float(loss.detach()),
            "data_cross_entropy": float(data_loss.detach()),
            "l2_penalty": float(penalty.detach()),
        }
        return loss

    started = time.perf_counter()
    optimizer.step(closure)
    diagnostics = {
        "backend": "torch.optim.LBFGS full-batch multinomial logistic regression",
        "formulation": "scikit/cuML-compatible multinomial L2 objective",
        "C": c_value,
        "feature_scaling": False,
        "max_iter": max_iter,
        "closure_calls": calls,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        **final_parts,
    }
    return weight.detach().cpu(), bias.detach().cpu(), diagnostics


def _directions_and_subspace(train_records: list[dict], rank: int = 4):
    means = torch.stack(
        [record["activations"].float().mean(1) for record in train_records]
    )
    pair_per_base = means[:, None, :, :] - means[:, :, None, :]
    pair_vectors = pair_per_base.mean(0)
    centered = means - means.mean(1, keepdim=True)
    _u, singular, vh = torch.linalg.svd(
        centered.reshape(-1, centered.shape[-1]), full_matrices=False
    )
    basis = vh[:rank].T.contiguous()
    projected = torch.einsum("std,dk,ek->ste", pair_vectors, basis, basis)
    norms = projected.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = projected / norms
    tool = ROLE_TO_INDEX["tool"]
    unit[tool, tool] = 0
    consistency = {}
    for source_index, source in enumerate(ROLES):
        if source == "tool":
            continue
        differences = pair_per_base[:, source_index, tool]
        cosine = F.cosine_similarity(
            differences, pair_vectors[source_index, tool][None, :], dim=-1
        )
        consistency[source] = {
            "mean": float(cosine.mean()),
            "std": float(cosine.std(unbiased=True)),
            "min": float(cosine.min()),
        }
    orthogonality_error = float((basis.T @ basis - torch.eye(rank)).abs().max())
    return (
        pair_vectors,
        basis,
        unit,
        {
            "rank": rank,
            "singular_values": singular[:rank].tolist(),
            "orthogonality_max_abs_error": orthogonality_error,
            "pair_direction_consistency": consistency,
            "raw_to_projected_norm": {
                role: {
                    "raw": float(pair_vectors[index, tool].norm()),
                    "projected": float(projected[index, tool].norm()),
                    "unit": float(unit[index, tool].norm()),
                }
                for index, role in enumerate(ROLES)
                if role != "tool"
            },
        },
    )


def analyze_v2_representation_activations(
    repo: Path,
    c_value: float = 5e-3,
    max_iter: int = 100,
) -> Path:
    root = repo / "research_outputs/phase3_steering_v2/repr_pre_mlp"
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("v2 pre-MLP activation collection is incomplete")
    output = repo / "research_outputs/phase3_steering_v2/representation_analysis"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "steering_state.pt"
    if state_path.exists():
        raise FileExistsError(f"{state_path} exists")
    state = {
        "schema_version": 1,
        "roles": list(ROLES),
        "role_to_index": ROLE_TO_INDEX,
        "hook": metadata["hook"],
        "upstream_name": metadata["upstream_name"],
        "layers": metadata["selected_layers"],
        "hidden_size": metadata["hidden_size"],
        "C": c_value,
        "temperature_scaling": False,
        "probe_weight": {},
        "probe_bias": {},
        "pair_vector": {},
        "role_basis": {},
        "unit_pair_direction": {},
    }
    reports = []
    started = time.perf_counter()
    for layer in metadata["selected_layers"]:
        layer_started = time.perf_counter()
        records = _load_layer_records(root, layer)
        train_x, train_y, train_weight, train_records = _split_features(
            records, "repr_train"
        )
        weight, bias, fit = _fit_upstream_style_probe(
            train_x, train_y, train_weight, c_value, max_iter
        )
        del train_x, train_y, train_weight
        pair, basis, unit, geometry = _directions_and_subspace(train_records)
        state["probe_weight"][layer] = weight
        state["probe_bias"][layer] = bias
        state["pair_vector"][layer] = pair
        state["role_basis"][layer] = basis
        state["unit_pair_direction"][layer] = unit
        split_metrics = {}
        for split in ("repr_dev", "repr_test"):
            x, y, sample_weight, selected = _split_features(records, split)
            metrics = _weighted_metrics(x, y, sample_weight, weight.cuda(), bias.cuda())
            metrics["underlying_bases"] = len(selected)
            split_metrics[split] = metrics
            del x, y, sample_weight
        report = {
            "layer": layer,
            "fit": fit,
            "repr_dev": split_metrics["repr_dev"],
            "repr_test": split_metrics["repr_test"],
            "geometry": geometry,
            "elapsed_seconds": round(time.perf_counter() - layer_started, 3),
        }
        reports.append(report)
        _atomic_json(
            output / "run_state.json",
            {
                "status": "running",
                "completed_layers": [item["layer"] for item in reports],
                "remaining_layers": [
                    item
                    for item in metadata["selected_layers"]
                    if item not in {done["layer"] for done in reports}
                ],
            },
        )
        print(
            f"v2 layer {layer}: dev={report['repr_dev']['base_balanced_accuracy']:.4f} "
            f"test={report['repr_test']['base_balanced_accuracy']:.4f} "
            f"fit={fit['elapsed_seconds']:.1f}s total={report['elapsed_seconds']:.1f}s",
            flush=True,
        )
        del records, train_records
        torch.cuda.empty_cache()
    ranked = sorted(
        reports,
        key=lambda item: item["repr_dev"]["base_balanced_accuracy"],
        reverse=True,
    )
    candidate_layers = [item["layer"] for item in ranked[:3]]
    state["candidate_layers_by_dev_accuracy"] = candidate_layers
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, state_path)
    report = {
        "schema_version": 1,
        "method": "paper-aligned pre-MLP five-way role probes",
        "selection_rule": "top three layers by repr_dev accuracy; repr_test not used",
        "candidate_layers_by_dev_accuracy": candidate_layers,
        "probe_sanity_pass": all(
            item["repr_test"]["base_balanced_accuracy"] > 0.5 for item in ranked[:3]
        ),
        "state_sha256": _sha256_file(state_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "layers": reports,
    }
    _atomic_json(output / "analysis_report.json", report)
    _atomic_json(
        output / "run_state.json",
        {
            "status": "complete",
            "completed_layers": metadata["selected_layers"],
            "remaining_layers": [],
        },
    )
    return output
