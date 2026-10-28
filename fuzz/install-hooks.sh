#!/usr/bin/env bash
# Install the git pre-commit hook that runs fuzz/guard.sh.
set -e
cd "$(dirname "$0")/.."
HOOK="$(git rev-parse --git-path hooks)/pre-commit"
cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
# Installed by fuzz/install-hooks.sh: refuse commits that break the guard rails.
exec "$(git rev-parse --show-toplevel)/fuzz/guard.sh"
EOF
chmod +x "$HOOK"
echo "installed $HOOK"
