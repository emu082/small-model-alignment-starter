import torch

def build_sft_batch(tokenizer, prompts, responses, max_length):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    all_ids, all_labels = [], []
    for p, r in zip(prompts, responses):
        p_ids = list(tokenizer.encode(p, add_special_tokens=False))
        r_ids = list(tokenizer.encode(r, add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            r_ids = r_ids + [tokenizer.eos_token_id]

        ids = (p_ids + r_ids)[:max_length]
        labels = ([-100] * len(p_ids) + r_ids)[:max_length]
        all_ids.append(ids)
        all_labels.append(labels)

    longest = max(len(x) for x in all_ids)
    input_ids, attention_mask, labels = [], [], []
    for ids, lab in zip(all_ids, all_labels):
        n_pad = longest - len(ids)
        input_ids.append(ids + [pad_id] * n_pad)
        attention_mask.append([1] * len(ids) + [0] * n_pad)
        labels.append(lab + [-100] * n_pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

def _response_logps_from_batch(model, batch):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()      # position t predicts token t+1
    labels = labels[:, 1:]                  # drop first label

    mask = labels != -100
    safe_labels = labels.masked_fill(~mask, 0)   # -100 is not a valid index
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask
    return token_logp.sum(dim=-1), mask.sum(dim=-1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    summed, _ = _response_logps_from_batch(model, batch)
    return summed

def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    summed, counts = _response_logps_from_batch(model, batch)
    total_tokens = counts.sum().clamp(min=1)
    return -summed.sum() / total_tokens
