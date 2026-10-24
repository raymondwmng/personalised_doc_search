#!/usr/bin/env bash
# Recipe to run the complete research benchmark for Representation, Tagging, and Search/Retrieval
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================================="
echo " Starting EUR-Lex Search & Retrieval Research Benchmark"
echo "=========================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Local Virtualenv: $PROJECT_ROOT/.venv"

# 1. Run Corpus Audit (Representation & Tagging distribution)
echo ""
echo "[Step 1/3] Auditing representation chunk lengths and tagging distribution..."
.venv/bin/python "$SCRIPT_DIR/audit_system.py"

# 2. Run Benchmark Recipe (Retrieval + Mistral-small-2603)
echo ""
echo "[Step 2/3] Running 9 benchmark queries across 3 search cases and evaluating downstream results..."
.venv/bin/python "$SCRIPT_DIR/benchmark_recipe.py"

# 3. Generate Visualizations and Notebook
echo ""
echo "[Step 3/3] Generating charts and analysis figures..."
.venv/bin/python "$SCRIPT_DIR/visualize_results.py"

echo ""
echo "=========================================================="
echo " Benchmark & Research Study Complete!"
echo " Outputs saved to: $SCRIPT_DIR/intermediate_outputs/"
echo " Figures saved to: $SCRIPT_DIR/figures/"
echo "=========================================================="
