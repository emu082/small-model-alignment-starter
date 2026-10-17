import os, json, math, random, time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from src import sft_core

MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LEN = 512
BS, ACCUM = 2, 16
LR, WD, CLIP, WARMUP = 2e-5, 0.1, 1.0, 0.03
EVAL_EVERY = 100
SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"
OUT = "outputs/sft_smoke" if SMOKE else "outputs/sft"

def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

def load_data():
    tr, va = "data/sft/sft_train_30k.json", "data/sft/sft_val_1k.json"
    if not (os.path.exists(tr) and os.path.exists(va)):
        from datasets import load_dataset
        raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        raw = raw.shuffle(seed=SEED).select(range(31000))
        pairs = [{"prompt": e["messages"][0]["content"], "response": e["messages"][1]["content"]} for e in raw]
        os.makedirs("data/sft", exist_ok=True)
        json.dump(pairs[:30000], open(tr, "w")); json.dump(pairs[30000:], open(va, "w"))
    return json.load(open(tr)), json.load(open(va))

@torch.no_grad()
def evaluate(model, tok, val):
    model.eval(); nll, n = 0.0, 0
    for i in range(0, len(val), 4):
        chunk = val[i:i + 4]
        batch = sft_core.build_sft_batch(tok, [format_prompt(x["prompt"]) for x in chunk], [x["response"] for x in chunk], MAX_LEN)
        with torch.autocast("cuda", dtype=torch.float16):
            s, c = sft_core._response_logps_from_batch(model, batch)
        nll += -s.sum().item(); n += c.sum().item()
    model.train()
    return nll / max(n, 1)

def save(model, tok, path):
    model.save_pretrained(path); tok.save_pretrained(path)

def main():
    if os.path.exists(f"{OUT}/final"):
        print("final checkpoint exists, skipping training"); return
    random.seed(SEED); torch.manual_seed(SEED); os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to("cuda:0")
    model.config.use_cache = False; model.train()
    train, val = load_data()
    random.Random(SEED).shuffle(train)
    if SMOKE: train, val = train[:64], val[:8]
    loader = DataLoader(train, batch_size=BS, shuffle=False, collate_fn=lambda b: b)
    total = len(loader) // ACCUM; warm = max(1, int(WARMUP * total))
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}], lr=LR)
    def lr_lambda(s):
        if s < warm: return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    eval_every = 2 if SMOKE else EVAL_EVERY
    best, step, micro, run_loss, t0 = 1e9, 0, 0, 0.0, time.time()
    log = open(f"{OUT}/metrics.jsonl", "a")
    for batch in loader:
        prompts = [format_prompt(x["prompt"]) for x in batch]; resps = [x["response"] for x in batch]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sft_core.compute_sft_loss(model, tok, prompts, resps, MAX_LEN)
        scaler.scale(loss / ACCUM).backward(); run_loss += loss.item() / ACCUM; micro += 1
        if micro % ACCUM == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            rec = {"step": step, "train_loss": run_loss, "lr": sched.get_last_lr()[0], "minutes": (time.time() - t0) / 60}
            run_loss = 0.0
            if step % eval_every == 0 or step == total:
                rec["val_loss"] = evaluate(model, tok, val)
                if rec["val_loss"] < best:
                    best = rec["val_loss"]; save(model, tok, f"{OUT}/best"); rec["saved_best"] = True
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
            if step >= total: break
    save(model, tok, f"{OUT}/final"); print("done. best val loss:", best)

if __name__ == "__main__":
    main()
