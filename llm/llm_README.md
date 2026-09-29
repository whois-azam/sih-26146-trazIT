# Local LLM Setup — `llm/` directory

This is the offline explanation-rephrasing/grounded-Q&A layer discussed for the dashboard. It **never** sees raw transactions and **never** makes a risk judgment — it only rephrases or answers questions using evidence the deterministic pipeline (Phase 6) already computed. See `query_llm.py`'s module docstring for the exact contract.

## Strict grounding (default, recommended — do not disable)

This is built as a **forensic evidence tool, not a general chatbot**, on purpose: an investigator needs to trust that anything the model says traces back to real evidence, and a general-purpose mode would reintroduce exactly the hallucination risk you're trying to eliminate. Four independent layers enforce this:

1. **Pre-filter** — refuses obviously off-topic questions before even calling the model (tested: correctly passes "why is the risk score high," correctly refuses "write me a poem" and "what's the capital of France").
2. **System prompt with concrete refusal examples** — small models follow demonstrated patterns far better than abstract "don't hallucinate" instructions.
3. **Stateless single-turn calls** — no conversation history, nothing can leak in from outside the one evidence dict passed to that call.
4. **Post-hoc numeric verification** — every number in the model's response is checked against numbers that actually appear in the evidence. Any untraceable number fails the whole response, which is replaced with a safe fallback message instead of shown to the user. Tested against both a fully-fabricated response and a subtly *partial* one (real risk score + one fake statistic) — caught both correctly.

If a model call keeps failing layer 4 in practice, that's a sign the model is drifting (or the evidence is too sparse for it to work with) — worth lowering `temperature` further or shortening the evidence list, not disabling the check.

## Directory layout

```
llm/
├── models/              <- put your .gguf file(s) here, nothing else needed
├── query_llm.py          (already built — auto-detects any .gguf in models/)
└── README.md             (this file)
```

You do not need to edit any code to add or swap a model — `query_llm.py` picks up the first `.gguf` file it finds in `llm/models/`.

## 1. Install the runtime

```bash
pip install llama-cpp-python --break-system-packages
```

On a normal machine this pulls a prebuilt wheel and takes seconds. (It timed out in my sandbox because no prebuilt wheel matched that exact environment and pip fell back to compiling from source — that shouldn't happen on your machine, but if it does: `sudo apt install cmake build-essential` first, then retry.)

## 2. Download a model (pick one)

**Recommended: Qwen3-1.7B-Instruct, Q4_K_M quantization (~1.1–1.4 GB)**

```bash
pip install huggingface_hub --break-system-packages
huggingface-cli download Qwen/Qwen3-1.7B-Instruct-GGUF \
    Qwen3-1.7B-Instruct-Q4_K_M.gguf \
    --local-dir llm/models --local-dir-use-symlinks False
```

**Smaller alternative if disk-constrained: Qwen3-0.6B (~400–500 MB)**

```bash
huggingface-cli download Qwen/Qwen3-0.6B-Instruct-GGUF \
    Qwen3-0.6B-Instruct-Q4_K_M.gguf \
    --local-dir llm/models --local-dir-use-symlinks False
```

If `huggingface-cli` isn't available or you prefer a browser: search the exact filenames above on huggingface.co, download manually, and drop the `.gguf` file straight into `llm/models/`.

## 3. Test it

```bash
cd llm
python3 query_llm.py          # confirms the model file is detected
```

Then from Python:

```python
from query_llm import EvidenceLLM
llm = EvidenceLLM()  # auto-finds the .gguf in models/
# example: entity_explanation is the dict from explainability/evidence.py's
# EntityExplanation.to_dict() (already produced by Phase 6)
print(llm.summarize_case_file(entity_explanation))
print(llm.answer_question("Why was this flagged?", entity_explanation))
```

## 4. If you want to compare 1.7B vs 0.6B before committing

Download both into `llm/models/` (the loader always picks the first alphabetically — rename one temporarily, or pass `model_path=` explicitly to `EvidenceLLM(model_path="llm/models/Qwen3-0.6B-Instruct-Q4_K_M.gguf")`), run the same evidence dict through both, and compare output quality on your own real explanations before deciding which one to keep.
