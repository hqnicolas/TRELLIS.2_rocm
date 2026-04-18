#!/usr/bin/env bash
set -euo pipefail

umask 0002
echo "Container started with umask 0002 for shared file access"

exec "$@"
