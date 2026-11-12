#!/usr/bin/env bash
# Download the input data (about 6.3 GB) into data/, as listed in data/MANIFEST.sha256.
# Files that are already there with the right checksum are skipped.
#
#   bash scripts/download_data.sh            # everything
#   bash scripts/download_data.sh pass8      # only the paths that contain "pass8"
#
# The files are the assets of a GitHub release of this repository; each asset is named
# as the file (no folders).  Set the address of that release here or in the environment.
set -euo pipefail

DATA_URL="${CRBREAK_DATA_URL:-https://github.com/fvalenciano/crbreak/releases/download/data-v1}"

cd "$(dirname "$0")/../data"
filter="${1:-}"

while read -r sum path; do
    [[ -n "${filter}" && "${path}" != *"${filter}"* ]] && continue
    if [[ -f "${path}" ]] && echo "${sum}  ${path}" | sha256sum --check --status; then
        echo "ok        ${path}"
        continue
    fi
    echo "download  ${path}"
    mkdir -p "$(dirname "${path}")"
    curl --fail --location --continue-at - --output "${path}" "${DATA_URL}/$(basename "${path}")"
    echo "${sum}  ${path}" | sha256sum --check --quiet
done < MANIFEST.sha256
echo "All files in place."
