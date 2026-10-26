#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

echo "=========================================================================="
echo "  Executing Empirical Study: Long-Context Embedding & Hierarchical Chunking"
echo "  Target Architecture: Section 3.2.2 (ModernBERT, Nomic, BGE-M3, Jina, UltraBERT)"
echo "=========================================================================="

"$PROJECT_ROOT/.venv/bin/python" -m experiments.long_context_embed_study.run_study "$@"

echo ""
echo "Study run complete. Results available in:"
echo "  - experiments/long_context_embed_study/intermediate_outputs/"
echo "  - experiments/long_context_embed_study/figures/"
echo "  - experiments/long_context_embed_study/RESEARCH_REPORT.md"
