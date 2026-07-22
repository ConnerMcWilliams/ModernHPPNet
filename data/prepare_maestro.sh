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
        exit 1
    fi
}

if [ -f "$ZIP" ]; then
    echo "$ZIP already present; skipping download."
elif command -v aria2c >/dev/null 2>&1; then
    echo "Downloading with aria2c ($JOBS connections) ..."
    aria2c -x "$JOBS" -s "$JOBS" -o "$ZIP" "$URL"
else
    echo "Downloading the MAESTRO dataset ..."
    download_ranged
fi

########################################################################
# 2. Extract in parallel
########################################################################
echo "Extracting the files with $JOBS workers ..."

# List archive entries, then extract them concurrently. unzip can extract a
# single named entry per invocation; bsdtar (libarchive) is the fallback.
if command -v unzip >/dev/null 2>&1; then
    list_cmd=(unzip -Z1 "$ZIP")
    extract_one() { unzip -o -q "$1" "$2"; }
elif command -v bsdtar >/dev/null 2>&1; then
    list_cmd=(bsdtar -tf "$ZIP")
    extract_one() { bsdtar -xf "$1" "$2"; }
else
    echo "Neither unzip nor bsdtar found; cannot extract." >&2
    exit 1
fi
export -f extract_one

"${list_cmd[@]}" \
    | grep -v '/$' \
    | xargs -d '\n' -P "$JOBS" -I {} bash -c 'extract_one "$0" "$1"' "$ZIP" {}

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
