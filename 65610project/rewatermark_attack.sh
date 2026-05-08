#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $0 <input.wav> [output-dir] [attacker-rounds]

Examples inside the Docker container:
  /audiowmark/65610project/rewatermark_attack.sh /data/in.wav
  /audiowmark/65610project/rewatermark_attack.sh in.wav
  /audiowmark/65610project/rewatermark_attack.sh /data/in.wav /data/rewatermark_attack_out 3

If <input.wav> is a relative path and does not exist in the current
directory, this script also tries /data/<input.wav>.

Optional:
  AUDIOWMARK=/path/to/audiowmark $0 <input.wav> [output-dir] [attacker-rounds]
EOF
}

find_audiowmark() {
  if [[ -n "${AUDIOWMARK:-}" ]]; then
    printf '%s\n' "$AUDIOWMARK"
    return
  fi

  if command -v audiowmark >/dev/null 2>&1; then
    command -v audiowmark
    return
  fi

  for candidate in ./audiowmark /usr/local/bin/audiowmark /audiowmark/audiowmark; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  echo "Error: could not find audiowmark. Set AUDIOWMARK=/path/to/audiowmark." >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

AUDIOWMARK_BIN="$(find_audiowmark)"
IN="$1"

if [[ ! -f "$IN" && "$IN" != /* && -f "/data/$IN" ]]; then
  IN="/data/$IN"
fi

if [[ ! -f "$IN" ]]; then
  echo "Error: input WAV not found: $1" >&2
  if [[ "$1" != /* ]]; then
    echo "Also checked: /data/$1" >&2
  fi
  exit 1
fi

if [[ $# -ge 2 ]]; then
  OUTDIR="$2"
elif [[ -d /data && -w /data ]]; then
  OUTDIR="/data/rewatermark_attack_out"
else
  OUTDIR="rewatermark_attack_out"
fi

ROUNDS="${3:-3}"
if ! [[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: attacker-rounds must be a positive integer." >&2
  exit 2
fi

mkdir -p "$OUTDIR"

VICTIM_KEY="$OUTDIR/victim.key"
ATTACKER_KEY="$OUTDIR/attacker.key"
VICTIM_AUDIO="$OUTDIR/audio_victim.wav"

M_VICTIM="11111111111111111111111111111111"
M_ATTACKER="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

echo "[1] generating victim and attacker keys"
"$AUDIOWMARK_BIN" gen-key "$VICTIM_KEY" --name "victim"
"$AUDIOWMARK_BIN" gen-key "$ATTACKER_KEY" --name "attacker"

echo "[2] embedding victim watermark"
"$AUDIOWMARK_BIN" add --key "$VICTIM_KEY" "$IN" "$VICTIM_AUDIO" "$M_VICTIM"

echo "[3] baseline: decoding victim-watermarked audio with victim key"
"$AUDIOWMARK_BIN" get --key "$VICTIM_KEY" "$VICTIM_AUDIO" | tee "$OUTDIR/decode_00_victim_audio_with_victim_key.txt"

echo "[4] baseline: decoding victim-watermarked audio with attacker key"
"$AUDIOWMARK_BIN" get --key "$ATTACKER_KEY" "$VICTIM_AUDIO" | tee "$OUTDIR/decode_00_victim_audio_with_attacker_key.txt"

PREV_AUDIO="$VICTIM_AUDIO"
for round in $(seq 1 "$ROUNDS"); do
  NEXT_AUDIO="$OUTDIR/audio_rewatermarked_${round}.wav"
  echo "[$((4 + round * 3 - 2))] attacker re-watermark round $round"
  "$AUDIOWMARK_BIN" add --key "$ATTACKER_KEY" "$PREV_AUDIO" "$NEXT_AUDIO" "$M_ATTACKER"

  echo "[$((4 + round * 3 - 1))] decoding round $round with victim key"
  "$AUDIOWMARK_BIN" get --key "$VICTIM_KEY" "$NEXT_AUDIO" | tee "$OUTDIR/decode_${round}_with_victim_key.txt"

  echo "[$((4 + round * 3))] decoding round $round with attacker key"
  "$AUDIOWMARK_BIN" get --key "$ATTACKER_KEY" "$NEXT_AUDIO" | tee "$OUTDIR/decode_${round}_with_attacker_key.txt"

  PREV_AUDIO="$NEXT_AUDIO"
done

echo
echo "Expected evidence:"
echo "  baseline victim decode should contain: $M_VICTIM"
echo "  baseline attacker decode should not contain: $M_ATTACKER"
echo "  after re-watermarking, victim decode shows whether the old watermark survived"
echo "  after re-watermarking, attacker decode should contain: $M_ATTACKER"
echo
echo "Useful files:"
echo "  baseline victim audio: $VICTIM_AUDIO"
echo "  final re-watermarked audio: $PREV_AUDIO"
echo "  decode logs: $OUTDIR/decode_*.txt"
echo
echo "Audiowmark binary: $AUDIOWMARK_BIN"
echo "Input audio: $IN"
echo "Outputs saved in: $OUTDIR"
