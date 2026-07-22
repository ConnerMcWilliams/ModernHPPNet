#! /bin/bash
#
# Parallelized MAESTRO v3.0.0 data pipeline: download -> unzip -> FLAC convert.
#
# Every stage that can be split across cores is. Tunables via env vars:
#   JOBS   number of parallel workers          (default: nproc)
#   URL    dataset url                          (default: MAESTRO v3.0.0)
#   ZIP    local zip filename                   (default: basename of URL)
#
# The download is parallelized with ranged HTTP GETs (the GCS bucket supports
# byte-range requests), or aria2c when available. Extraction and audio
# conversion fan out over $JOBS workers.

set -euo pipefail

JOBS="${JOBS:-$(nproc)}"
URL="${URL:-https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip}"
ZIP="${ZIP:-$(basename "$URL")}"
DIR="${ZIP%.zip}"

echo "Using $JOBS parallel workers."

# Pick the archive back end up front so the download skip-guard can reuse the
# lister to verify an existing zip before trusting it.
if command -v unzip >/dev/null 2>&1; then
    list_cmd=(unzip -Z1 "$ZIP")
    test_cmd=(unzip -t -qq "$ZIP")
    extract_many() { local zip="$1"; shift; unzip -o -q "$zip" "$@"; }
elif command -v bsdtar >/dev/null 2>&1; then
    list_cmd=(bsdtar -tf "$ZIP")
    test_cmd=(bsdtar -tf "$ZIP")
    extract_many() { local zip="$1"; shift; bsdtar -xf "$zip" "$@"; }
else
    echo "Neither unzip nor bsdtar found; cannot extract." >&2
    exit 1
fi
export -f extract_many

########################################################################
# 1. Download (101 GB) in parallel
########################################################################
download_ranged() {
    # Split the file into $JOBS contiguous byte ranges and fetch them
    # concurrently, then concatenate. Requires the server to honor Range
    # requests (GCS does).
    local size
    size=$(curl -sIL "$URL" | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {v=$2} END{gsub(/\r/,"",v); print v}')
    if [ -z "$size" ] || [ "$size" -le 0 ] 2>/dev/null; then
        echo "Could not determine file size; falling back to single-stream download."
        curl -fL -o "$ZIP" "$URL"
        return
    fi

    echo "Downloading $size bytes across $JOBS ranged connections ..."
    local chunk=$(( (size + JOBS - 1) / JOBS ))
    local pids=()
    for ((i = 0; i < JOBS; i++)); do
        local start=$(( i * chunk ))
        [ "$start" -ge "$size" ] && break
        local end=$(( start + chunk - 1 ))
        [ "$end" -ge "$size" ] && end=$(( size - 1 ))
        curl -fsL --retry 5 --retry-delay 2 -r "${start}-${end}" -o "${ZIP}.part${i}" "$URL" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done

    # Concatenate parts in order, then clean up.
    : > "$ZIP"
    for ((i = 0; i < JOBS; i++)); do
        [ -f "${ZIP}.part${i}" ] || continue
        cat "${ZIP}.part${i}" >> "$ZIP"
        rm -f "${ZIP}.part${i}"
    done

    local got
    got=$(stat -c '%s' "$ZIP")
    if [ "$got" -ne "$size" ]; then
        echo "Size mismatch: expected $size, got $got" >&2
        rm -f "$ZIP"
        exit 1
    fi
}

if [ -s "$ZIP" ] && "${test_cmd[@]}" >/dev/null 2>&1; then
    echo "$ZIP already present and valid; skipping download."
elif command -v aria2c >/dev/null 2>&1; then
    [ -f "$ZIP" ] && echo "$ZIP failed integrity check; re-downloading." >&2
    echo "Downloading with aria2c ($JOBS connections) ..."
    x="$JOBS"; [ "$x" -gt 16 ] && x=16
    aria2c -x "$x" -s "$JOBS" -o "$ZIP" "$URL"
else
    [ -f "$ZIP" ] && echo "$ZIP failed integrity check; re-downloading." >&2
    echo "Downloading the MAESTRO dataset ..."
    download_ranged
fi

########################################################################
# 2. Extract in parallel
########################################################################
echo "Extracting the files with $JOBS workers ..."

# List the file entries once, split them into $JOBS roughly-equal batches, and
# run one extractor per batch. Each invocation re-parses the central directory
# only once (instead of once per entry) and extracts all its members in a
# single call.
entries=$("${list_cmd[@]}" | grep -v '/$' || true)
if [ -z "$entries" ]; then
    echo "Archive contains no file entries; nothing to extract." >&2
    exit 1
fi
total=$(printf '%s\n' "$entries" | grep -c .)
batch=$(( (total + JOBS - 1) / JOBS ))
[ "$batch" -lt 1 ] && batch=1

# Pre-create every parent directory single-threaded FIRST. Parallel extractors
# otherwise race to mkdir shared parents; the loser aborts that entry with
# "checkdir error: ... File exists" and silently drops the file. Deriving dirs
# from the file paths (rather than trusting the archive's own dir entries) covers
# archives that omit them; mkdir -p handles nesting and pre-existing dirs.
printf '%s\n' "$entries" | sed 's:/[^/]*$::' | sort -u \
    | while IFS= read -r d; do [ -n "$d" ] && mkdir -p "$d"; done

printf '%s\n' "$entries" \
    | xargs -d '\n' -P "$JOBS" -n "$batch" bash -c 'extract_many "$0" "$@"' "$ZIP"

echo "Extraction done."

# rm "$ZIP"

########################################################################
# 3. Convert audio to FLAC in parallel
########################################################################
if command -v ffmpeg >/dev/null 2>&1; then
    echo "Converting the audio files to FLAC with $JOBS workers ..."
    convert_one() { ffmpeg -y -loglevel fatal -i "$1" -ac 1 -ar 16000 "${1%.wav}.flac"; }
    export -f convert_one
    find "$DIR" -name '*.wav' -print0 \
        | xargs -0 -P "$JOBS" -I {} bash -c 'convert_one "$0"' {}
else
    echo "ffmpeg not found; skipping FLAC conversion." >&2
fi

echo
echo "Preparation complete!"
