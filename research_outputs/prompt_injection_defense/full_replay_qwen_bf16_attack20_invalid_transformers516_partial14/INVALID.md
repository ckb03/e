# INVALID partial run

This partial 14-case run must not be used for metrics. It loaded both the gpt-oss target and Qwen judge under Transformers 5.16.1. None of the 14 completed cases matched the saved baseline on the pre-intervention target turn (0/14), so it violates the matched-condition protocol.

The valid replacement run is [`../full_replay_qwen_bf16_attack20/`](../full_replay_qwen_bf16_attack20/). It runs the target under the original Transformers 4.57.5 environment and isolates the direct Qwen BF16 judge behind a loopback-only HTTP process.
