Selection Stage: Role-Probe Training and Candidate Localization

The selection stage examines every newly returned Tool result and identifies a small number of token ranges that may contain prompt injection. It is a candidate generator rather than the final classifier: its selected ranges are passed to a separate language-model judge that decides whether they contain an untrusted instruction conflicting with the user task.

The role probe was trained on 250 unique ordinary-text samples: 188 from Dolma3 and 62 from the English C4 validation split. Texts were truncated to at most 1,024 GPT-OSS tokens and divided by underlying document into 175 training, 35 development, and 40 test samples. All five role-rendered versions of one text remained in the same split.

Each text was rendered five times with identical content tokens but different native GPT-OSS role framing: System, User, CoT, Assistant, and Tool. CoT used the assistant analysis channel, Assistant used the assistant final channel, and Tool used the functions-to-assistant commentary channel. The model processed the full framed sequence, but only activations aligned with the shared content tokens trained the probe, so the content stayed fixed while its role changed.

Activations were collected from openai/gpt-oss-20b immediately before each transformer block MLP, at the post-attention layer-normalization output. The hidden-state size is 2,880. Separate probes were trained at zero-based even layers 0 through 22.

At each evaluated layer, a five-class linear probe computes z_l(t) = W_l h_l(t) + b_l for token t, followed by a softmax over System, User, CoT, Assistant, and Tool. The probes used multinomial logistic regression with full-batch PyTorch L-BFGS, L2 regularization C = 0.005, a strong-Wolfe line search, and at most 100 iterations. No feature scaling or probability temperature scaling was applied.

Tokens were weighted so every underlying text contributed equally: for a text with n tokens, each token received weight 1/n. Since every text appeared under all five roles, the role classes also contributed equal total weight.

Layers were first ranked by five-way accuracy on the representation-development split. The strongest three even layers were layer 16 at 81.05 percent, layer 18 at 78.67 percent, and layer 8 at 77.20 percent. Their independent test accuracies were 80.87, 79.05, and 75.20 percent respectively.

A second attack-development analysis selected role-and-layer channels for localization rather than using overall probe accuracy alone. CoT-forgery injections produced the clearest sustained CoT signal at layer 8, while base injections were captured better by a User signal at layer 18. The deployed selector therefore uses CoT probability from layer 8 and User probability from layer 18.

At runtime, the detector is invoked for each new Tool result. It reconstructs the full conversation, isolates the new Tool-message tokens, runs GPT-OSS through layer 18, captures contextualized pre-MLP activations at layers 8 and 18, and applies the trained probe weights. In the HTML evaluation, it analyzes the serialized Tool result containing stdout, stderr, and exit code, with raw HTML inside stdout.

The detector uses local contrast instead of absolute role probability. For role r at layer l and position t, c_l,r(t) = mean64(p_l,r)(t) - mean512(p_l,r)(t). The centered 64-token mean measures sustained local role resemblance, while the centered 512-token mean estimates the surrounding Tool-result background.

A CoT seed is created when layer-8 CoT contrast is greater than 0.50 for at least eight consecutive tokenizer positions. A User seed is created when layer-18 User contrast is greater than 0.13 for at least eight consecutive positions. These conditions use OR, so either channel can independently create a seed. Each seed stores its token boundaries, channel, and peak contrast.

Every seed is padded by 64 tokens on both sides. Padded ranges are sorted and merged when the gap between them is no more than 64 tokens. A merged region retains its contributing channels and maximum CoT8 and User18 contrasts. Structural features are calculated over this region, and the emitted range receives another 64 tokens on each side, giving an isolated seed approximately 128 tokens of surrounding context.

The deployed selector uses vocabulary-free structural filtering and does not search for terms such as curl, upload, .env, ignore, or execute. It removes approximate HTML tags, decodes HTML entities, and computes word count, alphabetic fraction, markup fraction, word density, and natural-language mass. Natural-language mass equals word count multiplied by one minus markup fraction.

Any merged region containing a layer-8 CoT seed enters the CoT pool and bypasses the prose filter. CoT candidates are ranked by maximum CoT8 contrast, and the best two are retained. A CoT candidate is not reconsidered in the User pool even if it also contains User18 evidence.

A User-only region must contain at least 12 words, have alphabetic fraction at least 0.45, markup fraction no greater than 0.40, and word density at least 0.32 words per token. Passing User candidates are ranked by natural-language mass, and the best four are retained.

The selection stage returns zero to six candidate token ranges: up to two CoT candidates followed by up to four User candidates. It never adds filler candidates to reach the cap. Each output contains a start token, stop token, channel, and ranking score. The next stage expands these ranges toward semantic text or HTML boundaries before the deterministic Qwen judge evaluates them.
