#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# unlock_hour30.sh  --  run this on your instance when the password is given out
#
#     bash ~/challenge/unlock_hour30.sh <password>
#
# It opens the hold-out photographs into ~/challenge/testset and checks nothing
# was damaged on the way. Then score yourself exactly as you have been doing:
#
#     python ~/challenge/harness/evaluate.py \\
#       --submission my_submission.py \\
#       --images ~/challenge/testset/images \\
#       --annotations ~/challenge/testset/instances.json \\
#       --out my_result.json
set -euo pipefail
export SHLVL=0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$HERE/testset"
[ $# -eq 1 ] || { echo "usage: bash $0 <password>"; exit 2; }
[ -f "$HERE/hour30.enc" ] || { echo "hour30.enc is not next to this script"; exit 1; }

# The encrypted file's own checksum is deliberately not checked here. It changes
# every time the delivery is rebuilt, because the encryption is salted, and that
# made a rebuilt opener reject a file already copied to an instance. What matters
# is the checksum of the CONTENTS after decryption, below: that catches a wrong
# password and a damaged file alike.

echo "opening..."
rm -rf "$DST"; mkdir -p "$DST"
if ! openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 \
      -in "$HERE/hour30.enc" -pass "pass:$1" 2>/dev/null > "$DST/testset.tgz"; then
  rm -rf "$DST"; echo "that password did not work."; exit 1
fi
plain=$(md5sum "$DST/testset.tgz" | cut -d' ' -f1)
[ "$plain" = "ae97a0c23f6600995d2e2e0a9e1f5e62" ] || { rm -rf "$DST"; echo "wrong password, or damaged."; exit 1; }
tar xzf "$DST/testset.tgz" -C "$DST" && rm -f "$DST/testset.tgz"

n=$(find "$DST/images" -type f | wc -l | tr -d ' ')
[ "$n" = "800" ] || { echo "expected 800 photographs, found $n"; exit 1; }
echo "  $n photographs in $DST/images"

# Read every photograph once so they are in the page cache. Measured on a
# Wormhole n150 instance, 26 October 2026: the first run over these files took
# 236.4 ms a frame and the second took 84.9 ms, identical in every other
# respect. Without this the first team scored pays 121 seconds that later
# teams do not, which would make the ranking partly a matter of who went first.
echo "  warming the photographs so the first run is not penalised..."
find "$DST/images" -type f -exec cat {} + > /dev/null 2>&1 || true
echo "  warm"
echo "  answers in $DST/instances.json"
echo
echo "Score yourself with:"
echo "  python ~/challenge/harness/evaluate.py --submission YOUR_FILE.py \\"
echo "    --images ~/challenge/testset/images \\"
echo "    --annotations ~/challenge/testset/instances.json --out result.json"
