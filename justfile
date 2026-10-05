# List available recipes.
default:
    @just --list

# Check notebooks with marimo and lint with ruff.
check:
    #!/usr/bin/env bash
    set -euo pipefail
    files=$(find notebooks -name '*.py' -not -path '*/reference/*' -not -path '*/tests/*')
    if [ -n "$files" ]; then
        uvx marimo check --strict $files
        uv run --no-project python scripts/check_empty_cells.py $files
    fi
    uvx typos
    uvx ruff check notebooks templates

# Run a notebook headless in script mode.
smoke path:
    SMOKE=1 uv run --script {{path}}

# Open a notebook for interactive editing.
edit path:
    uvx marimo edit --sandbox {{path}}

# Export a notebook session (cell state) as JSON.
session path:
    uvx marimo export session --sandbox {{path}}

# Scaffold a new notebook folder from templates/notebook.py.
new slug:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="notebooks/{{slug}}"
    if [ -e "$dir" ]; then
        echo "error: $dir already exists" >&2
        exit 1
    fi
    snake=$(printf '%s' "{{slug}}" | tr '-' '_')
    title=$(printf '%s' "{{slug}}" | tr '-' ' ')
    mkdir -p "$dir"
    cp templates/notebook.py "$dir/$snake.py"
    printf '# %s\n\nStatus: planned\n\n- Molab:\n- Sources:\n\n## Files\n\n- `%s.py` — the entry notebook.\n' "$title" "$snake" > "$dir/README.md"

# List notebook folders and their status.
list:
    #!/usr/bin/env bash
    set -euo pipefail
    for readme in notebooks/*/README.md; do
        [ -e "$readme" ] || continue
        folder=$(dirname "$readme")
        status=$(grep -m1 '^Status:' "$readme" | sed 's/^Status:[[:space:]]*//')
        printf '%-42s %s\n' "$folder" "$status"
    done
