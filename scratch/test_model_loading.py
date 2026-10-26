import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

def test_model(model_name, **kwargs):
    print(f"\n--- Testing {model_name} ---")
    try:
        model = SentenceTransformer(model_name, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        emb = model.encode(["This is a test legal text."], normalize_embeddings=True)
        print(f"SUCCESS with SentenceTransformer! Shape: {emb.shape}, Dim: {emb.shape[1]}")
        return True
    except Exception as e:
        print(f"SentenceTransformer failed: {e}")
        try:
            print("Trying AutoModel / AutoTokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            inputs = tokenizer(["This is a test legal text."], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                out = model(**inputs)
            print(f"SUCCESS with AutoModel! Keys: {list(out.keys())}")
            return True
        except Exception as e2:
            print(f"AutoModel failed: {e2}")
            return False

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "answerdotai/ModernBERT-base"
    test_model(name, trust_remote_code=True)
