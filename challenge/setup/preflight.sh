#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# Preflight for a hackathon N150 instance. Read-only: changes nothing.
# Compares the instance against the reference the challenge was measured on
# (data/board.txt, data/env.txt, 2026-09-24).
REF_FW="19.11.0"; REF_KMD="2.9.0"; REF_METAL="9f9cd4fd590"
export PATH="$HOME/.local/bin:$PATH"
row() { printf '%-12s %-22s ref %-14s %s\n' "$1" "$2" "$3" "$4"; }
cmp() { [ "$1" = "?" ] && echo "UNKNOWN" || { [[ "$1" == "$2"* ]] && echo "match" || echo "DIFFERS"; }; }

KMD=$(cat /sys/module/tenstorrent/version 2>/dev/null || modinfo -F version tenstorrent 2>/dev/null || echo "?")
SNAP=$(tt-smi -s 2>/dev/null)
FW=$(printf '%s' "$SNAP" | python3 -c 'import json,sys
def find(o):
  if isinstance(o,dict):
    for k,v in o.items():
      if k == "fw_bundle_version" and isinstance(v,str): return v
      r=find(v)
      if r: return r
  elif isinstance(o,list):
    for v in o:
      r=find(v)
      if r: return r
try: print(find(json.load(sys.stdin)) or "?")
except Exception: print("?")')
BOARD=$(printf '%s' "$SNAP" | python3 -c 'import json,sys
try:
  d=json.load(sys.stdin)["device_info"][0]["board_info"]; print(d.get("board_type","?"))
except Exception: print("?")')
METAL=$(git -C "$HOME/tt-metal" rev-parse --short=11 HEAD 2>/dev/null || echo "?")

echo "host        $(hostname)  $(. /etc/os-release; echo "$PRETTY_NAME")"
row board "$BOARD" "n150 L" "$( [[ "$BOARD" == n150* ]] && echo match || echo DIFFERS)"
row firmware "$FW" "$REF_FW" "$(cmp "$FW" "$REF_FW")"
row kmd "$KMD" "$REF_KMD" "$(cmp "$KMD" "$REF_KMD")"
row tt-metal "$METAL" "$REF_METAL" "$(cmp "$METAL" "$REF_METAL")"
[ -f "$HOME/tt-metal/build/lib/_ttnncpp.so" ] && echo "built tt-metal  present" || echo "built tt-metal  MISSING (setup script will stop at STAGE 0)"
df -h "$HOME" | awk 'NR==2{print "disk free   ", $4}'

echo; echo "device open test (ttnn):"
( source "$HOME/tt-metal/python_env/bin/activate" 2>/dev/null
  cd "$HOME/tt-metal" && TT_METAL_HOME="$HOME/tt-metal" PYTHONPATH="$HOME/tt-metal" timeout 300 python - <<'PY'
import ttnn, torch
d = ttnn.open_device(device_id=0)
print("  grid", d.compute_with_storage_grid_size())
a = ttnn.from_torch(torch.ones(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
r = ttnn.to_torch(ttnn.add(a, a))
print("  add check", "PASS" if torch.allclose(r.float(), torch.full((32, 32), 2.0)) else "FAIL")
ttnn.close_device(d)
PY
) 2>&1 | grep -E 'grid|add check|Error|error|FATAL' | tail -5
