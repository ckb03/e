from __future__ import annotations

from pathlib import Path

from .defense import JUDGE_SYSTEM_PROMPT


def harmony_binary_prompt(prompt: str) -> str:
    """Wrap a binary classification request in the gpt-oss Harmony format."""
    return (
        f"<|start|>system<|message|>{JUDGE_SYSTEM_PROMPT}<|end|>"
        f"<|start|>user<|message|>{prompt}<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
    )


class GptOssTransformersJudge:
    """Deterministic, direct-Transformers binary judge for local smoke tests."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        cache_dir: Path | str = "/workspace/.hf_home/hub",
        device: str = "cuda:0",
        attn_implementation: str | None = "kernels-community/vllm-flash-attn3",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self.url = "direct-transformers"
        self.device = device
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            cache_dir=str(cache_dir),
            local_files_only=True,
            add_eos_token=False,
            add_bos_token=False,
        )
        model_kwargs = {
            "cache_dir": str(cache_dir),
            "local_files_only": True,
            "dtype": "auto",
            "device_map": device,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self._model = AutoModelForCausalLM.from_pretrained(
            model, **model_kwargs
        ).eval()
        self._finish_setup()

    @classmethod
    def from_loaded(
        cls,
        loaded_model,
        tokenizer,
        model_name: str = "openai/gpt-oss-20b",
        device: str = "cuda:0",
    ) -> GptOssTransformersJudge:
        """Reuse an already-loaded target model for sequential judge calls."""
        import torch

        judge = cls.__new__(cls)
        judge.model = model_name
        judge.url = "direct-transformers-shared-target"
        judge.device = device
        judge._torch = torch
        judge._tokenizer = tokenizer
        judge._model = loaded_model
        judge._finish_setup()
        return judge

    def _finish_setup(self) -> None:
        self._return_token = self._tokenizer.convert_tokens_to_ids("<|return|>")

    def __call__(self, prompt: str) -> str:
        rendered = harmony_binary_prompt(prompt)
        inputs = self._tokenizer(rendered, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                eos_token_id=self._return_token,
                use_cache=True,
            )
        generated = output[0, inputs.input_ids.shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


class QwenBf16TransformersJudge:
    """Deterministic Qwen judge using cached FP8 weights dequantized to BF16."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3.8-27B-FP8",
        cache_dir: Path | str = "/workspace/.hf_home/hub",
        device: str = "cuda:0",
        attn_implementation: str = "eager",
    ) -> None:
        import time

        import torch
        if attn_implementation == "sdpa":
            torch.backends.cuda.enable_cudnn_sdp(False)
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            Qwen3_5ForConditionalGeneration,
        )
        from transformers.integrations.finegrained_fp8 import FP8Experts

        self.model = model
        self.url = "direct-transformers-bf16-dequantized"
        self.device = device
        self._torch = torch
        started = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        config = AutoConfig.from_pretrained(
            model,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        skip_modules = config.quantization_config["modules_to_not_convert"]
        removed_gate_skips = [
            name for name in skip_modules if name.endswith(".mlp.gate")
        ]
        config.quantization_config["modules_to_not_convert"] = [
            name for name in skip_modules if not name.endswith(".mlp.gate")
        ]
        config.quantization_config["dequantize"] = True
        FP8Experts._impl_tp_layer_overrides.setdefault(None, {})
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model,
            config=config,
            cache_dir=str(cache_dir),
            local_files_only=True,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        ).eval()
        self.load_seconds = time.perf_counter() - started
        print(
            f"Qwen judge loaded in {self.load_seconds:.3f}s after removing "
            f"{len(removed_gate_skips)} colliding mlp.gate skip entries",
            flush=True,
        )

    def __call__(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()
