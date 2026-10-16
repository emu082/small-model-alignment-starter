import os, json, random, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from src import sft_core, dpo_core

SFT_BEST = os.environ.get(
    "SFT_BEST",
    "/kaggle/input/notebooks/emaadharoon/notebook5e46fcbfb5/small-model-alignment-starter/outputs/sft/best")
DATASET = os.environ.get("DATASET", "hh")          # "hh" or "uf"
SMOKE = os.environ.get("SMOKE") == "1"
BETA, LR, ACCUM, CLIP = 0.1, 1e-6, 32, 1.0
MAX_LEN, MAX_PROMPT = 512, 256
EVAL_EVERY = 1 if SMOKE else 100
SEED = 42
OUT = f"outputs/dpo_{DATASET}" + ("_smoke" if SMOKE else "")


def format_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def make_prompt(tok, instr):
    ids = tok.encode(instr, add_special_tokens=False)
    limit = MAX_PROMPT - 24          # room for the template tokens
    if len(ids) > limit:
        instr = tok.decode(ids[:limit])
    return format_prompt(instr)


def save(model, tok, path):
    model.save_pretrained(path)
    tok.save_pretrained(path)


@torch.no_grad()
def evaluate(policy, ref, tok, val):
    policy.eval()
    losses, accs, margins = [], [], []
    for i in range(0, len(val), 2):
        chunk = val[i:i + 2]
        batch = dpo_core.build_dpo_batch(
            tok, [make_prompt(tok, x["prompt"]) for x in chunk],
            [x["chosen"] for x in chunk], [x["rejected"] for x in chunk], MAX_LEN)
        with torch.autocast("cuda", dtype=torch.float16):
            pc, _ = sft_core._response_logps_from_batch(policy, batch["chosen"])
            pr, _ = sft_core._response_logps_from_batch(policy, batch["rejected"])
        rc, _ = sft_core._response_logps_from_batch(ref, batch["chosen"])
        rr, _ = sft_core._response_logps_from_batch(ref, batch["rejected"])
        rc, rr = rc.to(pc.device), rr.to(pc.device)
        rw, rl = BETA * (pc - rc), BETA * (pr - rr)
        losses += (-F.logsigmoid(rw - rl)).tolist()
        accs += (rw > rl).float().tolist()
        margins += (rw - rl).tolist()
    policy.train()
    n = len(losses)
    return {"val_loss": sum(losses) / n, "val_reward_acc": sum(accs) / n,
            "val_reward_margin": sum(margins) / n}


def main():
    if os.path.exists(f"{OUT}/final"):
        print("final checkpoint exists, skipping")
        return
    random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    prefix = "hh" if DATASET == "hh" else "uf"
    train = json.load(open(f"data/dpo/{prefix}_train_15k.json"))
    val = json.load(open(f"data/dpo/{prefix}_val_1k.json"))
    if SMOKE:
        train, val = train[:64], val[:8]

    tok = AutoTokenizer.from_pretrained(SFT_BEST)
    policy = AutoModelForCausalLM.from_pretrained(SFT_BEST, dtype=torch.float32).to("cuda:0")
    policy.config.use_cache = False
    policy.train()
    ref = AutoModelForCausalLM.from_pretrained(SFT_BEST, dtype=torch.float16).to("cuda:1")
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    total = len(train) // ACCUM
    best, step, micro, run_loss, t0 = (-1.0, 1e9), 0, 0, 0.0, time.time()
    log = open(f"{OUT}/metrics.jsonl", "a")

    for ex in train:
        with torch.autocast("cuda", dtype=torch.float16):
            loss = dpo_core.compute_dpo_loss(
                policy, ref, tok, [make_prompt(tok, ex["prompt"])],
                [ex["chosen"]], [ex["rejected"]], BETA, MAX_LEN)
        scaler.scale(loss / ACCUM).backward()
        run_loss += loss.item() / ACCUM
        micro += 1

        if micro % ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "train_loss": run_loss, "minutes": (time.time() - t0) / 60}
            run_loss = 0.0
            if step % EVAL_EVERY == 0 or step == total:
                rec.update(evaluate(policy, ref, tok, val))
                key = (rec["val_reward_acc"], -rec["val_loss"])
                if key > (best[0], -best[1]):
                    best = (rec["val_reward_acc"], rec["val_loss"])
                    save(policy, tok, f"{OUT}/best")
                    rec["saved_best"] = True
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
            if step >= total:
                break

    save(policy, tok, f"{OUT}/final")
    print("done. best val reward acc:", best[0])


if __name__ == "__main__":
    main()
