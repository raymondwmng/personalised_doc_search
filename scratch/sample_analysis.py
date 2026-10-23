import sys
from pathlib import Path
import pandas as pd
import json
from collections import Counter
import random

data_dir = Path("data/eurlex_kaggle")
csv_path = data_dir / "EurLex_all.csv"

print(f"Reading columns and first few rows from {csv_path}...")
df_head = pd.read_csv(csv_path, nrows=5)
print("Columns in EurLex_all.csv:")
for c in df_head.columns:
    print(f" - {c} (example: {repr(str(df_head[c].iloc[0])[:80])})")

# Let's count total lines or sample documents across chunks
chunk_size = 5000
sample_target = 2000
sampled_rows = []

print("\nSampling documents across the dataset...")
# Read with chunksize, picking a sample from each chunk to ensure wide coverage across decades and types
reader = pd.read_csv(
    csv_path,
    chunksize=chunk_size,
    low_memory=False
)

total_seen = 0
for i, chunk in enumerate(reader):
    total_seen += len(chunk)
    # take sample from chunk
    # e.g., if total dataset is ~140,000 rows, ~28 chunks of 5000. Take ~75 rows per chunk.
    n_sample = min(len(chunk), 100)
    chunk_sample = chunk.sample(n=n_sample, random_state=42 + i)
    sampled_rows.append(chunk_sample)
    if i % 5 == 0:
        print(f"Processed chunk {i}, total rows seen: {total_seen}...")

sample_df = pd.concat(sampled_rows, ignore_index=True)
print(f"\nTotal rows scanned in dataset: {total_seen}")
print(f"Total sampled documents: {len(sample_df)}")

# Analysis
print("\n--- Summary Analysis of Sampled Documents ---")
print("\nAct Types:")
print(sample_df["Act_type"].value_counts(dropna=False).head(10))

print("\nDate Range:")
dates = pd.to_datetime(sample_df["Date_document"], errors="coerce")
print(f"Min Date: {dates.min()}")
print(f"Max Date: {dates.max()}")
print("\nYear Distribution (decades):")
sample_df["Year"] = dates.dt.year
print(sample_df["Year"].dropna().astype(int).groupby(lambda y: (y // 10) * 10).count())

# Text length stats
if "act_raw_text" in sample_df.columns:
    text_lengths_chars = sample_df["act_raw_text"].dropna().astype(str).apply(len)
    text_lengths_words = sample_df["act_raw_text"].dropna().astype(str).apply(lambda t: len(t.split()))
    print("\nText Length (Characters):")
    print(text_lengths_chars.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))
    print("\nText Length (Words):")
    print(text_lengths_words.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

# EUROVOC analysis
print("\n--- EUROVOC Terms Analysis ---")
eurovoc_counter = Counter()
docs_with_eurovoc = 0
for val in sample_df["EUROVOC"].dropna():
    val_str = str(val).strip()
    if not val_str or val_str.lower() == "nan":
        continue
    docs_with_eurovoc += 1
    # Check delimiter in EUROVOC (comma, semicolon, or newline?)
    # Common format might be semicolon separated or comma separated
    terms = [t.strip() for t in val_str.replace(";", ",").split(",") if t.strip()]
    eurovoc_counter.update(terms)

print(f"Docs with EUROVOC: {docs_with_eurovoc} / {len(sample_df)} ({docs_with_eurovoc/len(sample_df):.1%})")
print("Top 25 EUROVOC Descriptors:")
for term, count in eurovoc_counter.most_common(25):
    print(f"  {term}: {count}")

# Subject Matter analysis
print("\n--- Subject Matter Analysis ---")
subj_counter = Counter()
docs_with_subj = 0
for val in sample_df["Subject_matter"].dropna():
    val_str = str(val).strip()
    if not val_str or val_str.lower() == "nan":
        continue
    docs_with_subj += 1
    terms = [t.strip() for t in val_str.replace(";", ",").split(",") if t.strip()]
    subj_counter.update(terms)

print(f"Docs with Subject_matter: {docs_with_subj} / {len(sample_df)} ({docs_with_subj/len(sample_df):.1%})")
print("Top 25 Subject Matters:")
for term, count in subj_counter.most_common(25):
    print(f"  {term}: {count}")

# Let's save a structured sample of diverse topics with title, celex, eurovoc, subject matter, and brief text snippet
samples_by_topic = []
topics_of_interest = [
    "agriculture", "environment", "competition", "trade", "transport",
    "energy", "fisheries", "health", "consumer", "intellectual property",
    "social policy", "monetary", "customs"
]

print("\n--- Sample Document Examples Across Diverse Domains ---")
examples_found = {}
for idx, row in sample_df.iterrows():
    euro = str(row.get("EUROVOC", "")).lower()
    subj = str(row.get("Subject_matter", "")).lower()
    text = str(row.get("act_raw_text", ""))
    title = str(row.get("Act_name", ""))
    celex = str(row.get("CELEX", ""))
    act_type = str(row.get("Act_type", ""))
    date = str(row.get("Date_document", ""))

    combined_meta = f"{euro} {subj}".lower()
    for t in topics_of_interest:
        if t not in examples_found and (t in combined_meta):
            snippet = text[:300].replace("\n", " ").strip()
            examples_found[t] = {
                "topic": t,
                "celex": celex,
                "act_type": act_type,
                "date": date,
                "title": title,
                "eurovoc": str(row.get("EUROVOC", ""))[:120],
                "subject_matter": str(row.get("Subject_matter", ""))[:120],
                "snippet": snippet
            }

for topic, ex in examples_found.items():
    print(f"\n[Domain: {topic.upper()}]")
    print(f"CELEX: {ex['celex']} | Type: {ex['act_type']} | Date: {ex['date']}")
    print(f"Title: {ex['title']}")
    print(f"Subject: {ex['subject_matter']}")
    print(f"EUROVOC: {ex['eurovoc']}")
    print(f"Snippet: {ex['snippet'][:200]}...")

# Also check codebook if possible
try:
    import pypdf
    pdf_path = data_dir / "CEPS_EurLex_codebook.pdf"
    if pdf_path.exists():
        reader = pypdf.PdfReader(str(pdf_path))
        print(f"\n--- Codebook PDF Info ---")
        print(f"Codebook Pages: {len(reader.pages)}")
        codebook_text = ""
        for p in reader.pages[:3]:
            codebook_text += p.extract_text() + "\n"
        print("Codebook extract (first 1000 chars):")
        print(codebook_text[:1000])
except Exception as e:
    print(f"Codebook read note: {e}")
