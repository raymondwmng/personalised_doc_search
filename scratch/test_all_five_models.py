import sys
import types
import torch

# 1. Compatibility shim: transformers.onnx
onnx_mod = types.ModuleType("transformers.onnx")
class OnnxConfig:
    pass
onnx_mod.OnnxConfig = OnnxConfig
sys.modules["transformers.onnx"] = onnx_mod

# 2. Compatibility shim: transformers.pytorch_utils.find_pruneable_heads_and_indices
import transformers.pytorch_utils as pu
if not hasattr(pu, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index
    pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

# 3. Compatibility shim: PreTrainedModel.get_extended_attention_mask and PretrainedConfig defaults
import transformers.configuration_utils
_orig_getattr = getattr(transformers.configuration_utils.PretrainedConfig, "__getattr__", None)
def _config_getattr(self, key):
    defaults = {
        "is_decoder": False,
        "add_cross_attention": False,
        "pruned_heads": set(),
        "chunk_size_feed_forward": 0,
    }
    if key in defaults:
        return defaults[key]
    if _orig_getattr:
        return _orig_getattr(self, key)
    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
transformers.configuration_utils.PretrainedConfig.__getattr__ = _config_getattr

import transformers.modeling_utils
if not hasattr(transformers.modeling_utils.PreTrainedModel, "get_extended_attention_mask"):
    def get_extended_attention_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask {attention_mask.shape}")
        dt = dtype or getattr(self, "dtype", torch.float32)
        extended_attention_mask = extended_attention_mask.to(dtype=dt)
        return (1.0 - extended_attention_mask) * torch.finfo(dt).min
    transformers.modeling_utils.PreTrainedModel.get_extended_attention_mask = get_extended_attention_mask

if not hasattr(transformers.modeling_utils.PreTrainedModel, "get_head_mask"):
    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            if hasattr(self, "_convert_head_mask_to_5d"):
                head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask
    transformers.modeling_utils.PreTrainedModel.get_head_mask = get_head_mask

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing all 5 models on device: {device} (PyTorch {torch.__version__}, Transformers {transformers.__version__})")

MODELS = [
    ("ModernBERT-base", "nomic-ai/modernbert-embed-base", "sentence_transformers", {}),
    ("nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5", "sentence_transformers", {"trust_remote_code": True}),
    ("bge-m3", "BAAI/bge-m3", "sentence_transformers", {}),
    ("jina-embeddings-v2-base-en", "jinaai/jina-embeddings-v2-base-en", "sentence_transformers", {"trust_remote_code": True, "model_kwargs": {"low_cpu_mem_usage": False}}),
    ("UltraBERT-Embed-150M", "dmis-lab/UltraBERT-Embed-150M", "ultrabert", {}),
]

for name, model_id, loader_type, kwargs in MODELS:
    print(f"\n==========================================")
    print(f"Loading {name} ({model_id})...")
    try:
        if loader_type == "sentence_transformers":
            st_model = SentenceTransformer(model_id, device=device, **kwargs)
            emb = st_model.encode(["Sample EU legal excerpt for verification."], normalize_embeddings=True)
            print(f"SUCCESS {name}! Shape: {emb.shape}, Dim: {emb.shape[1]}")
        elif loader_type == "ultrabert":
            tok = AutoTokenizer.from_pretrained(model_id, extra_special_tokens={}, trust_remote_code=True)
            mod = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
            inp = tok(["Instruct: Retrieve EU law\nQuery: 470-790 MHz"], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                out = mod(**inp)
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            emb = torch.nn.functional.normalize(last_hidden.mean(dim=1), p=2, dim=-1)
            print(f"SUCCESS {name}! Shape: {emb.shape}, Dim: {emb.shape[1]}")
    except Exception as e:
        import traceback
        print(f"FAILED {name}: {e}")
        traceback.print_exc()

print("\nAll model tests finished.")
