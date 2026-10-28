import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from cu.data.cwru import load_cwru
from cu.data.ecg import load_ecg

CACHE = Path(os.environ.get("CU_CACHE_DIR", "cache")) / "data"
SUMS = Path(__file__).resolve().parents[1] / "configs" / "data_checksums.json"

for fid in ("130", "105"):
    A, Y, _, m = load_cwru(dict(file=fid, n_sig=1024, tau_ms=0.5, lam=0.05, eps_rel=1e-2), 5, 0, "train", allow_synthetic=False)
    print(f"CWRU {fid}: source {m['source']}, resonance {m['f_res']:.0f} Hz")
for split in ("train", "test"):
    A, Y, _, m = load_ecg(dict(dictionary="db4", measurement="gaussian", m_frac=0.5, snr_db=20.0, lam=0.02, eps_rel=1e-2), 5, 0, split)
    print(f"ECG {split}: {Y.shape[0]} windows loaded")

sums = {}
for f in [CACHE / "cwru" / f"{fid}.mat" for fid in ("105", "130")]:
    sums[f"cwru/{f.name}"] = hashlib.sha256(f.read_bytes()).hexdigest()
for f in sorted(f for f in (CACHE / "ecg").glob("*.npy") if "nstdb_em" not in f.name):
    a = np.ascontiguousarray(np.load(f), dtype=np.float64)
    sums[f"ecg/{f.name}"] = hashlib.sha256(a.tobytes()).hexdigest()
if "--record" in sys.argv:
    SUMS.write_text(json.dumps(sums, indent=1) + "\n")
    print("recorded", len(sums), "checksums in", SUMS)
else:
    ref = json.loads(SUMS.read_text())
    bad = [k for k in ref if k in sums and sums[k] != ref[k]]
    missing = [k for k in ref if k not in sums]
    if bad or missing:
        print("CHECKSUM MISMATCH:", bad, "MISSING:", missing)
        sys.exit(1)
    print(f"all {len(ref)} data files match the recorded checksums")
