#!/usr/bin/env python3
"""Serve the direct Qwen BF16 judge through a minimal local HTTP endpoint."""

from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from eval_harness.local_judge import QwenBf16TransformersJudge


def build_app(judge: QwenBf16TransformersJudge) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model": judge.model,
            "backend": judge.url,
            "load_seconds": judge.load_seconds,
        }

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> dict:
        body = await request.json()
        messages = body.get("messages", [])
        user_messages = [
            message.get("content", "")
            for message in messages
            if message.get("role") == "user"
        ]
        if not user_messages:
            raise HTTPException(status_code=400, detail="missing user message")
        if body.get("use_request_messages"):
            inputs = judge._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors="pt", return_dict=True,
            ).to(judge.device)
            with judge._torch.inference_mode():
                output = judge._model.generate(
                    **inputs, max_new_tokens=int(body.get("max_tokens", 512)),
                    do_sample=False, use_cache=True,
                )
            generated = output[0, inputs["input_ids"].shape[1]:]
            verdict = judge._tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip()
        else:
            verdict = judge(str(user_messages[-1]))
        return {
            "id": "qwen-bf16-local-judge",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": verdict},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--cache-dir", default="/workspace/.hf_home/hub")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="eager")
    args = parser.parse_args()

    judge = QwenBf16TransformersJudge(
        model=args.model,
        cache_dir=args.cache_dir,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    uvicorn.run(
        build_app(judge),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
