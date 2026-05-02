#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $0 <input.wav> [output-dir]

Examples inside the Docker container:
  /audiowmark/65610project/multikey_attack.sh /data/in.wav
  /audiowmark/65610project/multikey_attack.sh in.wav
  /audiowmark/65610project/multikey_attack.sh /data/in.wav /data/multikey_attack_out

If <input.wav> is a relative path and does not exist in the current
directory, this script also tries /data/<input.wav>.

Optional:
  AUDIOWMARK=/path/to/audiowmark $0 <input.wav> [output-dir]
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
  OUTDIR="/data/multikey_attack_out"
else
  OUTDIR="multikey_attack_out"
fi

mkdir -p "$OUTDIR"

KEY_A="$OUTDIR/key_A.key"
KEY_B="$OUTDIR/key_B.key"
AUDIO_A="$OUTDIR/audio_A.wav"
AUDIO_AB="$OUTDIR/audio_AB.wav"

M_A="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
M_B="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

echo "[1] generating keys"
"$AUDIOWMARK_BIN" gen-key "$KEY_A" --name "key_A"
"$AUDIOWMARK_BIN" gen-key "$KEY_B" --name "key_B"

echo "[2] embedding message A with key A"
"$AUDIOWMARK_BIN" add --key "$KEY_A" "$IN" "$AUDIO_A" "$M_A"

echo "[3] sanity check: decoding audio_A with key A"
"$AUDIOWMARK_BIN" get --key "$KEY_A" "$AUDIO_A" | tee "$OUTDIR/decode_audio_A_with_key_A.txt"

echo "[4] sanity check: decoding audio_A with key B"
"$AUDIOWMARK_BIN" get --key "$KEY_B" "$AUDIO_A" | tee "$OUTDIR/decode_audio_A_with_key_B.txt"

echo "[5] embedding message B with key B on top of audio_A"
"$AUDIOWMARK_BIN" add --key "$KEY_B" "$AUDIO_A" "$AUDIO_AB" "$M_B"

echo "[6] decoding final audio with key A"
"$AUDIOWMARK_BIN" get --key "$KEY_A" "$AUDIO_AB" | tee "$OUTDIR/decode_key_A.txt"

echo "[7] decoding final audio with key B"
"$AUDIOWMARK_BIN" get --key "$KEY_B" "$AUDIO_AB" | tee "$OUTDIR/decode_key_B.txt"

echo "[8] decoding final audio with both keys"
"$AUDIOWMARK_BIN" get --key "$KEY_A" --key "$KEY_B" "$AUDIO_AB" | tee "$OUTDIR/decode_both_keys.txt"



echo
echo "Expected evidence:"
echo "  audio_A with key_A should contain: $M_A"
echo "  audio_A with key_B should not contain: $M_B"
echo "  key_A decode should contain: $M_A"
echo "  key_B decode should contain: $M_B"
echo
echo "Audiowmark binary: $AUDIOWMARK_BIN"
echo "Input audio: $IN"
echo "Outputs saved in: $OUTDIR"
