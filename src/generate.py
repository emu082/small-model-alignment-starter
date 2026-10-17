import json, os, sys, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def format_prompt(x):
    return f"### Instruction:\n{x}\n\n### Response:\n"

def generate(ckpt, eval_path, out_path, name, dataset, bs=16, max_new=256):
    if os.path.exists(out_path):
        print("exists, skipping", out_path); return
    data = json.load(open(eval_path))
    for i, c in enumerate(data):
        c["_p"] = c.get("instruction") or c.get("prompt")
        c["_id"] = c.get("id", f"{dataset}_{i}")
    tok = AutoTokenizer.from_pretrained(ckpt); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float16).to("cuda:0").eval()
    res = []
    for i in tqdm(range(0, len(data), bs)):
        chunk = data[i:i+bs]
        enc = tok([format_prompt(c["_p"]) for c in chunk], return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for c, o in zip(chunk, out):
            text = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            res.append({"id": c["_id"], "instruction": c["_p"], "output": text, "generator": name, "dataset": dataset})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=2)
    print("wrote", out_path, len(res))

if __name__ == "__main__":
    ckpt, eval_path, out_path, name, dataset = sys.argv[1:6]
    generate(ckpt, eval_path, out_path, name, dataset)
