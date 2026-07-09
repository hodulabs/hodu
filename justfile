default:
    @just --list

# format rust + justfile
format: format-rs format-just

# format rust crates
format-rs:
    cargo fmt --all

# format the justfile (just --fmt is still unstable)
format-just:
    just --fmt --unstable

# verify formatting (rust + justfile), no writes
format-check:
    cargo fmt --all --check
    just --fmt --check --unstable

# lint rust + markdown
lint: lint-rs lint-md

# clippy over all targets, warnings denied
lint-rs:
    cargo clippy --workspace --all-targets -- -D warnings

# every relative link in the readme resolves to a real file (external urls skipped)
lint-md:
    #!/usr/bin/env bash
    set -euo pipefail
    status=0
    for md in README.md hodu/README.md hodu_core/README.md; do
        dir=$(dirname "$md")
        while read -r link; do
            case "$link" in http://*|https://*|mailto:*|'#'*) continue ;; esac
            path="${link%%#*}"
            [ -z "$path" ] && continue
            [ -e "$dir/$path" ] || { echo "BROKEN: $md -> $link"; status=1; }
        done < <(grep -oE '\]\([^)]+\)' "$md" | sed -E 's/^\]\(//; s/\)$//')
    done
    [ "$status" -eq 0 ] && echo "markdown links ok"
    exit $status

# workspace test suite (the Metal tests skip when no device is present)
test:
    cargo test --workspace

# build the workspace
build:
    cargo build --workspace

# regenerate third-party attribution (LICENSES/ + THIRD-PARTY.md) via cargo-tribute
licenses:
    cargo tribute

# fail if a dependency license is disallowed or the attribution is stale
licenses-check:
    cargo tribute --check

# CI gate: format check, lint, test
check: format-check lint test
