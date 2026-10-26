import json

with open("experiments/long_context_embed_study/intermediate_outputs/study_metrics.json") as f:
    metrics = json.load(f)

print("=" * 110)
print(f"{'Model':<26} | {'Regime':<6} | {'P@1':<5} | {'Case1':<6} | {'Case2':<6} | {'GT/5':<5} | {'MRR':<5} | {'C3 Peak':<7} | {'VRAM(GB)':<8} | {'Throughput':<12}")
print("=" * 110)

for m in metrics:
    key = m["model_key"]
    reg = m["regime"]
    p1 = m["overall_p1"]
    c1 = m["case1_singularity"]["p_at_1"]
    c2 = m["case2_multi_match"]["p_at_5"]
    gt = m["case2_multi_match"]["mean_gt_recovered_in_top5"]
    mrr = m["overall_mrr"]
    c3 = m["case3_zero_match"]["mean_peak_score"]
    prof = m.get("encoding_profile", {})
    vram = prof.get("peak_vram_gb", 0.0)
    docs_sec = prof.get("throughput_docs_per_sec", 0.0)
    toks_sec = prof.get("throughput_tokens_per_sec", 0.0)
    print(f"{key:<26} | {reg:<6} | {p1:<5.2f} | {c1:<6.2f} | {c2:<6.2f} | {gt:<5.2f} | {mrr:<5.3f} | {c3:<7.3f} | {vram:7.2f}  | {docs_sec:5.1f} docs/s")
