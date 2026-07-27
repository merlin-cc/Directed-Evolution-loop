#!/usr/bin/env bash
# Export a Jupyter notebook to HTML or PDF using this project's .venv,
# bypassing VS Code's flaky in-app export (which can reach for the wrong
# Python interpreter and fail trying to build pyzmq from source).
#
# Usage:
#   ./export_notebook.sh path/to/notebook.ipynb            # -> HTML (default)
#   ./export_notebook.sh path/to/notebook.ipynb pdf         # -> PDF (first run installs a headless Chromium, one-time)
#
# Exports the notebook as currently saved on disk -- it does NOT re-run cells,
# so whatever plots/outputs are already saved in the .ipynb are what you get.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUPYTER="$SCRIPT_DIR/.venv/bin/jupyter"

if [ $# -lt 1 ]; then
    echo "Usage: $0 path/to/notebook.ipynb [html|pdf]" >&2
    exit 1
fi

NOTEBOOK="$1"
FORMAT="${2:-html}"

if [ "$FORMAT" = "pdf" ]; then
    "$JUPYTER" nbconvert --to webpdf --allow-chromium-download "$NOTEBOOK"
else
    "$JUPYTER" nbconvert --to html "$NOTEBOOK"
fi
