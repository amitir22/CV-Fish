#!/usr/bin/env bash
root=${1:-.}

find "$root" -type f -name '*.py' -not -path '*/venv/*' -print0 \
| while IFS= read -r -d '' file; do
    grep -nH -E '^[[:space:]]*(import|from)[[:space:]]' "$file"
done

