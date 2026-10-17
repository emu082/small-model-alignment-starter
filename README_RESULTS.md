# Results and reproduction notes
Base: Qwen/Qwen2.5-0.5B. Prompt template: `### Instruction:\n{q}\n\n### Response:\n`.
Generation: greedy, max 256 new tokens, no repetition penalty, fp16, same for SFT and DPO.
Judge: google/gemini-3.6-flash (Kaggle model proxy), batches of 5, 5 s sleep, 3 retries.

| Track | Judged | DPO wins | SFT wins | Ties | Win rate excl. ties |
|---|---|---|---|---|---|
| UF DPO vs SFT (AlpacaEval 300, seed 42) | 295 (5 failed) | 93 | 122 | 80 | 0.433 |
| HH DPO vs SFT (HH eval 300) | 300 | 133 | 103 | 64 | 0.564 |

LoRA persona (12 prompts, run 4: lr 1e-4, 3 epochs, DPO + SFT term): style adapter 3 / base 0 / tie 9;
quality adapter 3 / base 8 / tie 1. Runs 1-3 (lr 5e-5/3ep, 2e-4/5ep, 1e-4/2ep DPO only) are documented in the report.

Training: SFT best val loss 1.3790 (937 steps). HH DPO best val reward acc 0.635. UF DPO best val reward acc 0.694 (step 200).
Scripts: src/train_sft.py, src/train_dpo.py, src/generate.py, src/train_lora.py, src/gen_persona.py, scripts/judge_pairwise.py.
Adapter tests: PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py (7 passed).
