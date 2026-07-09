"""CPU (gloo) TP smoke for qwen3_5_moe: reproduce the k_proj shard failure with the
shipped plan when world_size > num_key_value_heads, then validate a custom plan that
keeps attention un-head-sharded (colwise_gather_output) and shards the MoE experts.

Run:
  MODE=save    python tp_smoke.py                       # build tiny ckpt + reference logits
  MODE=auto    torchrun --nproc_per_node=2 tp_smoke.py  # expect reshape failure
  MODE=custom  torchrun --nproc_per_node=2 tp_smoke.py  # expect success + logits match + 2x backward
"""
import os, sys

# transformers (and its pure-python deps) come from the project venv via PYTHONPATH;
# torch 2.5.1+cpu comes from this scratch venv.
import torch
import torch.distributed as dist

MODE = os.environ.get("MODE", "save")
RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_q35moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM


def tiny_cfg():
    return Qwen3_5MoeTextConfig(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=4,           # layer_types default: full attn every 4th
        num_attention_heads=4,
        num_key_value_heads=1,         # 1 kv head < world_size 2 -> shipped plan must break
        head_dim=32,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=8,
        num_experts_per_tok=2,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )


CUSTOM_PLAN = {
    # full attention: weight-shard + gather output (works for any kv-head count)
    "model.layers.*.self_attn.q_proj": "colwise_gather_output",
    "model.layers.*.self_attn.k_proj": "colwise_gather_output",
    "model.layers.*.self_attn.v_proj": "colwise_gather_output",
    "model.layers.*.self_attn.o_proj": "colwise_gather_output",
    # linear attention: same as shipped
    "model.layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
    "model.layers.*.linear_attn.in_proj_z": "colwise_gather_output",
    "model.layers.*.linear_attn.in_proj_b": "colwise_gather_output",
    "model.layers.*.linear_attn.in_proj_a": "colwise_gather_output",
    "model.layers.*.linear_attn.out_proj": "colwise_gather_output",
    # MoE experts: true TP
    "model.layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "model.layers.*.mlp.experts.down_proj": "rowwise",
    "model.layers.*.mlp.experts": "moe_tp_experts",
    "model.layers.*.mlp.shared_expert.gate_proj": "colwise",
    "model.layers.*.mlp.shared_expert.up_proj": "colwise",
    "model.layers.*.mlp.shared_expert.down_proj": "rowwise",
    "lm_head": "colwise_gather_output",
}


def main():
    torch.manual_seed(0)
    if MODE == "save":
        cfg = tiny_cfg()
        model = Qwen3_5MoeForCausalLM(cfg).eval()
        model.save_pretrained(CKPT)
        ids = torch.arange(2 * 16).reshape(2, 16) % 256
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        torch.save({"ids": ids, "logits": logits}, os.path.join(CKPT, "ref.pt"))
        print("SAVED tiny ckpt + reference logits", logits.shape)
        return

    plan = "auto" if MODE == "auto" else CUSTOM_PLAN
    try:
        model = Qwen3_5MoeForCausalLM.from_pretrained(CKPT, tp_plan=plan, dtype=torch.float32).eval()
    except Exception as e:
        print(f"[rank{RANK}] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        raise
    ref = torch.load(os.path.join(CKPT, "ref.pt"), weights_only=True)
    ids = ref["ids"]
    try:
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        err = (logits - ref["logits"]).abs().max().item()
        print(f"[rank{RANK}] FORWARD OK  max|logits - ref| = {err:.3e}", flush=True)
    except Exception as e:
        print(f"[rank{RANK}] FORWARD FAILED: {type(e).__name__}: {e}", flush=True)
        if MODE == "auto":
            print(f"[rank{RANK}] (expected failure for shipped plan with kv_heads < world_size)", flush=True)
            return
        raise
    if MODE == "custom":
        # retain_graph + repeated backward through TP collectives (jlens-style)
        emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
        out = model(inputs_embeds=emb).logits
        s1 = out[..., :8].sum()
        s2 = out[..., 8:16].sum()
        s1.backward(retain_graph=True)
        g1 = emb.grad.clone(); emb.grad = None
        s2.backward()
        g2 = emb.grad.clone()
        print(f"[rank{RANK}] DOUBLE BACKWARD OK |g1|={g1.norm():.4e} |g2|={g2.norm():.4e}", flush=True)


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
