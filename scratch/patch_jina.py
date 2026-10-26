import glob

files = glob.glob('/home/speechray/.cache/huggingface/modules/transformers_modules/**/modeling_bert.py', recursive=True)
print(f"Found {len(files)} files")
for path in files:
    with open(path, 'r') as f:
        content = f.read()
    
    target = "alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position"
    if target in content and "relative_position.is_meta" not in content:
        replacement = """if relative_position.is_meta:
            self._current_alibi_size = size
            return torch.empty(1, n_heads, size, size, device='meta')
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position"""
        content = content.replace(target, replacement)
        with open(path, 'w') as f:
            f.write(content)
        print(f"Patched: {path}")
    else:
        print(f"Already patched or target not found: {path}")
