import torch
import torch.nn.functional as F
from src import sft_core


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    chosen = sft_core.build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected = sft_core.build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    return {"chosen": chosen, "rejected": rejected}


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                                reference_chosen_logps, reference_rejected_logps, beta):
    pi_gap = policy_chosen_logps - policy_rejected_logps
    ref_gap = reference_chosen_logps - reference_rejected_logps
    logits = beta * (pi_gap - ref_gap)
    return -F.logsigmoid(logits).mean()

def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)

    pc, _ = sft_core._response_logps_from_batch(policy_model, batch["chosen"])
    pr, _ = sft_core._response_logps_from_batch(policy_model, batch["rejected"])

    with torch.no_grad():
        rc, _ = sft_core._response_logps_from_batch(reference_model, batch["chosen"])
        rr, _ = sft_core._response_logps_from_batch(reference_model, batch["rejected"])

    return compute_dpo_loss_from_logps(pc, pr, rc.to(pc.device), rr.to(pc.device), beta)
