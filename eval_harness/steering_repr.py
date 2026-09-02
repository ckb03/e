from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .steering_data import ROLES

ROLE_TO_INDEX = {role: index for index, role in enumerate(ROLES)}
SCHEMA_VERSION = 1


def evenly_spaced_offsets(length: int, limit: int) -> list[int]:
    if length <= 0 or limit <= 0:
        raise ValueError("length and limit must be positive")
    if limit == 1:
        return [length // 2]
    if length <= limit:
        return list(range(length))
    return [index * (length - 1) // (limit - 1) for index in range(limit)]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class ResidualCapture:
    def __init__(self, layers) -> None:
        self.positions: torch.Tensor | None = None
        self.values: list[torch.Tensor | None] = [None] * len(layers)
        self.handles = [
            layer.register_forward_hook(self._make_hook(index))
            for index, layer in enumerate(layers)
        ]

    def _make_hook(self, layer_index: int) -> Callable:
        def hook(_module, _inputs, output) -> None:
            if self.positions is None:
                raise RuntimeError("capture positions were not configured")
            hidden = output[0] if isinstance(output, tuple) else output
            positions = self.positions.to(hidden.device)
            batches = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            selected = hidden[batches, positions]
            self.values[layer_index] = selected.detach().to(
                device="cpu",
                dtype=torch.bfloat16,
            )

        return hook

    def reset(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = [None] * len(self.values)

    def stacked(self) -> torch.Tensor:
        if any(value is None for value in self.values):
            missing = [
                index for index, value in enumerate(self.values) if value is None
            ]
            raise RuntimeError(f"missing captures for layers {missing}")
        return torch.stack(self.values)  # type: ignore[arg-type]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _load_local_model(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        cache_dir=config.cache_dir,
        local_files_only=True,
        add_eos_token=False,
        add_bos_token=False,
        padding_side="left",
    )
    model_kwargs = {
        "cache_dir": config.cache_dir,
        "local_files_only": True,
        "dtype": config.dtype,
        "device_map": config.device,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        **model_kwargs,
    ).eval()
    return tokenizer, model


def _validate_repr_manifest(rows: list[dict]) -> None:
    if len(rows) != 150:
        raise ValueError(f"expected 150 representation bases, found {len(rows)}")
    if len({row["base_id"] for row in rows}) != len(rows):
        raise ValueError("representation base IDs are not unique")
    for row in rows:
        if set(row["role_variants"]) != set(ROLES):
            raise ValueError(f"base {row['base_id']} does not contain all roles")
        for role in ROLES:
            variant = row["role_variants"][role]
            if (
                variant["content_token_end"] - variant["content_token_start"]
                != row["content_token_count"]
            ):
                raise ValueError(
                    f"base {row['base_id']} role {role} has misaligned content bounds"
                )


def collect_representation_activations(
    repo: Path,
    config_path: Path,
    max_tokens_per_base: int = 64,
    resume: bool = False,
) -> Path:
    manifest_path = repo / "eval_data/steering_repr_manifest.jsonl"
    rows = _read_jsonl(manifest_path)
    _validate_repr_manifest(rows)
    config = Config.load(config_path)
    output_dir = repo / "research_outputs/phase3_steering/repr_activations"
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest_path.relative_to(repo)),
        "manifest_sha256": _sha256_file(manifest_path),
        "model_id": config.model_id,
        "model_config_fingerprint": config.fingerprint(),
        "hook": "decoder block output before next block",
        "roles": list(ROLES),
        "max_tokens_per_base": max_tokens_per_base,
        "sampling": "deterministic evenly spaced content offsets",
        "sequence_weighting": "equal base sequence weight in downstream fits",
        "git_commit": _git_commit(repo),
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        invariant_keys = (
            "schema_version",
            "manifest_sha256",
            "model_id",
            "model_config_fingerprint",
            "hook",
            "roles",
            "max_tokens_per_base",
            "sampling",
            "sequence_weighting",
        )
        mismatches = [
            key for key in invariant_keys if previous.get(key) != metadata.get(key)
        ]
        if mismatches:
            raise ValueError(f"activation resume metadata mismatch: {mismatches}")
        if not resume:
            raise FileExistsError(f"{output_dir} exists; pass --resume")
        metadata = previous
    else:
        _atomic_json(metadata_path, metadata)

    completed = {int(path.stem.split("-")[-1]) for path in shard_dir.glob("base-*.pt")}
    unknown = completed - {int(row["base_id"]) for row in rows}
    if unknown:
        raise ValueError(f"unknown activation shard base IDs: {sorted(unknown)}")
    pending = [row for row in rows if int(row["base_id"]) not in completed]
    state_path = output_dir / "run_state.json"
    _atomic_json(
        state_path,
        {
            "status": "running" if pending else "complete",
            "completed_base_ids": sorted(completed),
            "remaining_base_ids": [int(row["base_id"]) for row in pending],
        },
    )
    if not pending:
        return output_dir

    tokenizer, model = _load_local_model(config)
    layers = model.model.layers
    metadata.update(
        {
            "num_layers": len(layers),
            "hidden_size": model.config.hidden_size,
            "model_type": type(model).__name__,
        }
    )
    _atomic_json(metadata_path, metadata)
    capture = ResidualCapture(layers)
    started = time.perf_counter()
    try:
        for row in pending:
            base_id = int(row["base_id"])
            prompts = [row["role_variants"][role]["prompt"] for role in ROLES]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            sequence_length = encoded.input_ids.shape[1]
            offsets = evenly_spaced_offsets(
                int(row["content_token_count"]),
                max_tokens_per_base,
            )
            positions = []
            for role_index, role in enumerate(ROLES):
                variant = row["role_variants"][role]
                prompt_length = int(encoded.attention_mask[role_index].sum().item())
                left_padding = sequence_length - prompt_length
                role_positions = [
                    left_padding + int(variant["content_token_start"]) + offset
                    for offset in offsets
                ]
                observed = encoded.input_ids[role_index, role_positions].tolist()
                expected = [row["content_token_ids"][offset] for offset in offsets]
                if observed != expected:
                    raise AssertionError(
                        f"runtime token alignment failed for base {base_id}, role {role}"
                    )
                positions.append(role_positions)
            position_tensor = torch.tensor(positions, dtype=torch.long)
            capture.reset(position_tensor)
            model_inputs = {
                key: value.to(config.device) for key, value in encoded.items()
            }
            case_started = time.perf_counter()
            with torch.inference_mode():
                model.model(**model_inputs, use_cache=False)
            torch.cuda.synchronize()
            activations = capture.stacked()
            shard = {
                "schema_version": SCHEMA_VERSION,
                "base_id": base_id,
                "split": row["split"],
                "roles": list(ROLES),
                "content_sha256": row["content_sha256"],
                "content_token_count": row["content_token_count"],
                "sample_offsets": offsets,
                "activations": activations,
            }
            final_path = shard_dir / f"base-{base_id:03d}.pt"
            temporary = final_path.with_suffix(".pt.tmp")
            torch.save(shard, temporary)
            os.replace(temporary, final_path)
            completed.add(base_id)
            remaining = [
                int(item["base_id"])
                for item in rows
                if int(item["base_id"]) not in completed
            ]
            _atomic_json(
                state_path,
                {
                    "status": "running" if remaining else "complete",
                    "completed_base_ids": sorted(completed),
                    "remaining_base_ids": remaining,
                },
            )
            print(
                f"base {base_id}: {len(offsets)} tokens x {len(ROLES)} roles x "
                f"{len(layers)} layers in {time.perf_counter() - case_started:.3f}s",
                flush=True,
            )
    finally:
        capture.close()

    metadata["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    metadata["shard_count"] = len(completed)
    metadata["status"] = "complete"
    _atomic_json(metadata_path, metadata)
    return output_dir


def _load_activation_shards(output_dir: Path) -> list[dict]:
    metadata = json.loads((output_dir / "metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("representation activation collection is incomplete")
    paths = sorted((output_dir / "shards").glob("base-*.pt"))
    if len(paths) != 150:
        raise ValueError(f"expected 150 activation shards, found {len(paths)}")
    records = [
        torch.load(path, map_location="cpu", weights_only=False) for path in paths
    ]
    if [record["base_id"] for record in records] != list(range(150)):
        raise ValueError("activation shard base IDs are not exactly 0..149")
    expected_shape = (metadata["num_layers"], len(ROLES), metadata["hidden_size"])
    for record in records:
        shape = record["activations"].shape
        if shape[0] != expected_shape[0] or shape[1] != expected_shape[1]:
            raise ValueError(
                f"invalid activation shape for base {record['base_id']}: {shape}"
            )
        if shape[3] != expected_shape[2] or shape[2] != len(record["sample_offsets"]):
            raise ValueError(
                f"invalid activation/token shape for base {record['base_id']}"
            )
    return records


def _split_layer_data(records: list[dict], split: str, layer: int):
    selected = [record for record in records if record["split"] == split]
    features = []
    labels = []
    weights = []
    for record in selected:
        values = record["activations"][layer].float()
        token_count = values.shape[1]
        features.append(values.reshape(-1, values.shape[-1]))
        labels.append(
            torch.arange(len(ROLES), dtype=torch.long).repeat_interleave(token_count)
        )
        weights.append(torch.full((len(ROLES) * token_count,), 1.0 / token_count))
    return (
        torch.cat(features),
        torch.cat(labels),
        torch.cat(weights),
        selected,
    )


def _weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    return (losses * weights).sum() / weights.sum()


def _fit_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_weight: torch.Tensor,
    dev_x: torch.Tensor,
    dev_y: torch.Tensor,
    dev_weight: torch.Tensor,
    seed: int,
    l2: float,
    epochs: int,
) -> tuple[torch.Tensor, torch.Tensor, float, dict]:
    device = torch.device("cuda")
    feature_mean = (train_x * train_weight[:, None]).sum(0) / train_weight.sum()
    centered = train_x - feature_mean
    feature_variance = (centered.square() * train_weight[:, None]).sum(
        0
    ) / train_weight.sum()
    feature_scale = feature_variance.sqrt().clamp_min(1e-4)
    train_z = ((train_x - feature_mean) / feature_scale).to(device)
    dev_z = ((dev_x - feature_mean) / feature_scale).to(device)
    train_y = train_y.to(device)
    dev_y = dev_y.to(device)
    train_weight = train_weight.to(device)
    dev_weight = dev_weight.to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    linear = torch.nn.Linear(train_z.shape[1], len(ROLES), device=device)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=0.03, weight_decay=0.0)
    best = None
    best_loss = math.inf
    patience = 0
    batch_size = 4096
    for epoch in range(epochs):
        permutation = torch.randperm(train_z.shape[0], generator=generator)
        for start in range(0, len(permutation), batch_size):
            index = permutation[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = linear(train_z[index])
            loss = (
                _weighted_cross_entropy(
                    logits,
                    train_y[index],
                    train_weight[index],
                )
                + 0.5 * l2 * linear.weight.square().sum()
            )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            dev_loss = float(
                _weighted_cross_entropy(linear(dev_z), dev_y, dev_weight).item()
            )
        if dev_loss < best_loss - 1e-5:
            best_loss = dev_loss
            best = {
                "weight": linear.weight.detach().cpu().clone(),
                "bias": linear.bias.detach().cpu().clone(),
                "epoch": epoch + 1,
            }
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    if best is None:
        raise RuntimeError("probe optimization produced no checkpoint")
    standardized_weight = best["weight"]
    raw_weight = standardized_weight / feature_scale[None, :]
    raw_bias = best["bias"] - raw_weight @ feature_mean
    with torch.no_grad():
        dev_logits = dev_x.to(device) @ raw_weight.to(device).T + raw_bias.to(device)
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    temp_optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        temp_optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = _weighted_cross_entropy(
            dev_logits / temperature,
            dev_y,
            dev_weight,
        )
        loss.backward()
        return loss

    temp_optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0).item())
    diagnostics = {
        "best_epoch": best["epoch"],
        "dev_cross_entropy_before_temperature": best_loss,
        "temperature": temperature,
        "l2": l2,
    }
    return raw_weight, raw_bias, temperature, diagnostics


def _probe_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    temperature: float,
) -> dict:
    logits = (features @ weight.T + bias) / temperature
    predictions = logits.argmax(-1)
    correct = predictions.eq(labels)
    weighted_accuracy = float((correct * weights).sum() / weights.sum())
    confusion = torch.zeros((len(ROLES), len(ROLES)), dtype=torch.int64)
    weighted_confusion = torch.zeros((len(ROLES), len(ROLES)), dtype=torch.float64)
    for target in range(len(ROLES)):
        for predicted in range(len(ROLES)):
            mask = labels.eq(target) & predictions.eq(predicted)
            confusion[target, predicted] = mask.sum()
            weighted_confusion[target, predicted] = weights[mask].sum()
    weighted_confusion /= weighted_confusion.sum(1, keepdim=True)
    return {
        "token_accuracy": float(correct.float().mean()),
        "base_balanced_accuracy": weighted_accuracy,
        "cross_entropy": float(_weighted_cross_entropy(logits, labels, weights)),
        "confusion_matrix_counts": confusion.tolist(),
        "confusion_matrix_base_balanced_rows": weighted_confusion.tolist(),
    }


def _pair_and_geometry(
    train_records: list[dict],
    layer: int,
    svd_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    hidden_size = train_records[0]["activations"].shape[-1]
    pair_per_base = []
    center_per_base = []
    covariance = torch.zeros((hidden_size, hidden_size), device="cuda")
    for record in train_records:
        values = record["activations"][layer].float().to("cuda")
        means = values.mean(1)
        pair_per_base.append(means[None, :, :] - means[:, None, :])
        center_per_base.append(values.mean((0, 1)).cpu())
        centered = values - values.mean(0, keepdim=True)
        flat = centered.reshape(-1, hidden_size)
        covariance.add_(flat.T @ flat / values.shape[1])
    pair_stack = torch.stack(pair_per_base)
    pair_vectors = pair_stack.mean(0).cpu()
    center = torch.stack(center_per_base).mean(0)
    covariance /= len(train_records)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    basis = eigenvectors[:, order[:svd_rank]].cpu()
    singular_values = eigenvalues.sqrt().cpu()
    tool_index = ROLE_TO_INDEX["tool"]
    consistency = {}
    for source_index, source in enumerate(ROLES):
        if source == "tool":
            continue
        differences = pair_stack[:, source_index, tool_index].cpu()
        direction = pair_vectors[source_index, tool_index]
        cosine = F.cosine_similarity(differences, direction[None, :], dim=-1)
        consistency[source] = {
            "mean": float(cosine.mean()),
            "std": float(cosine.std(unbiased=True)),
            "min": float(cosine.min()),
            "p10": float(torch.quantile(cosine, 0.10)),
            "median": float(torch.quantile(cosine, 0.50)),
            "p90": float(torch.quantile(cosine, 0.90)),
            "max": float(cosine.max()),
        }
    total_variance = eigenvalues.sum().clamp_min(1e-12)
    geometry = {
        "top_singular_values": singular_values[:svd_rank].tolist(),
        "top_explained_variance_ratio": (
            eigenvalues[:svd_rank] / total_variance
        ).tolist(),
        "cumulative_explained_variance_ratio": torch.cumsum(
            eigenvalues[:svd_rank] / total_variance,
            dim=0,
        ).tolist(),
        "pairwise_tool_direction_cosine": consistency,
    }
    return pair_vectors, basis, singular_values, center, geometry


def analyze_representation_activations(
    repo: Path,
    seed: int = 20260904,
    l2: float = 1e-4,
    epochs: int = 80,
    svd_rank: int = 16,
) -> Path:
    activation_dir = repo / "research_outputs/phase3_steering/repr_activations"
    records = _load_activation_shards(activation_dir)
    metadata = json.loads((activation_dir / "metadata.json").read_text())
    num_layers = int(metadata["num_layers"])
    hidden_size = int(metadata["hidden_size"])
    output_dir = repo / "research_outputs/phase3_steering/representation_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "run_state.json"
    state_file = output_dir / "steering_state.pt"
    if state_file.exists():
        raise FileExistsError(f"{state_file} already exists")
    probe_weights = torch.empty((num_layers, len(ROLES), hidden_size))
    probe_biases = torch.empty((num_layers, len(ROLES)))
    temperatures = torch.empty(num_layers)
    pair_vectors = torch.empty((num_layers, len(ROLES), len(ROLES), hidden_size))
    bases = torch.empty((num_layers, hidden_size, svd_rank))
    singular_values = torch.empty((num_layers, hidden_size))
    centers = torch.empty((num_layers, hidden_size))
    layer_reports = []
    started = time.perf_counter()
    for layer in range(num_layers):
        layer_started = time.perf_counter()
        train_x, train_y, train_weight, train_records = _split_layer_data(
            records, "repr_train", layer
        )
        dev_x, dev_y, dev_weight, _ = _split_layer_data(records, "repr_dev", layer)
        test_x, test_y, test_weight, _ = _split_layer_data(records, "repr_test", layer)
        weight, bias, temperature, fit = _fit_probe(
            train_x,
            train_y,
            train_weight,
            dev_x,
            dev_y,
            dev_weight,
            seed + layer,
            l2,
            epochs,
        )
        pair, basis, singular, center, geometry = _pair_and_geometry(
            train_records,
            layer,
            svd_rank,
        )
        metrics = _probe_metrics(
            test_x,
            test_y,
            test_weight,
            weight,
            bias,
            temperature,
        )
        probe_weights[layer] = weight
        probe_biases[layer] = bias
        temperatures[layer] = temperature
        pair_vectors[layer] = pair
        bases[layer] = basis
        singular_values[layer] = singular
        centers[layer] = center
        report = {
            "layer": layer,
            "hook": f"block_{layer}_output",
            "probe": {**metrics, **fit},
            "geometry": geometry,
            "elapsed_seconds": round(time.perf_counter() - layer_started, 3),
        }
        layer_reports.append(report)
        _atomic_json(
            state_path,
            {
                "status": "running",
                "completed_layers": list(range(layer + 1)),
                "remaining_layers": list(range(layer + 1, num_layers)),
            },
        )
        print(
            f"layer {layer}: test acc={metrics['base_balanced_accuracy']:.4f} "
            f"CE={metrics['cross_entropy']:.4f} "
            f"in {report['elapsed_seconds']:.3f}s",
            flush=True,
        )
    state = {
        "schema_version": SCHEMA_VERSION,
        "roles": list(ROLES),
        "role_to_index": ROLE_TO_INDEX,
        "hook": metadata["hook"],
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "svd_rank": svd_rank,
        "seed": seed,
        "l2": l2,
        "epochs_max": epochs,
        "repr_manifest_sha256": metadata["manifest_sha256"],
        "activation_metadata_sha256": _sha256_file(activation_dir / "metadata.json"),
        "probe_weight": probe_weights,
        "probe_bias": probe_biases,
        "probe_temperature": temperatures,
        "pair_vector": pair_vectors,
        "role_basis": bases,
        "singular_values": singular_values,
        "global_center": centers,
    }
    temporary = state_file.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, state_file)
    report = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "seed": seed,
            "l2": l2,
            "epochs_max": epochs,
            "svd_rank": svd_rank,
            "token_sampling": metadata["sampling"],
            "sequence_weighting": metadata["sequence_weighting"],
            "train_split": "repr_train",
            "temperature_split": "repr_dev",
            "reported_probe_split": "repr_test",
        },
        "state_sha256": _sha256_file(state_file),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "layers": layer_reports,
    }
    _atomic_json(output_dir / "analysis_report.json", report)
    _atomic_json(
        state_path,
        {
            "status": "complete",
            "completed_layers": list(range(num_layers)),
            "remaining_layers": [],
        },
    )
    return output_dir
