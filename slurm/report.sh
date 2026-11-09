#!/bin/bash
# Tables and progress for one experiment; called at the end of every job, or by hand:
#   bash slurm/report.sh configs/v2/stream_q15.yaml "seed=0"
# Each experiment writes its tables to <run folder>/paper/results (never into paper/ directly).
CONFIG="$1"; EXTRA="${2:-}"
read -r EXP_DIR NTASK < <(python -c "
import os, sys; sys.path.insert(0, '.')
from twcore.config import load_config
c = load_config('${CONFIG}', '${EXTRA}'.split())
d = os.path.join(c['out_dir'], c['exp']) + (f\"/s{int(c['seed'])}\" if c.get('seed_dirs') else '')
print(d, len(c['stream']))")
ROOT="${EXP_DIR%/s[0-9]*}"   # the experiment folder above the seed folders (v2 layout)
if [ "${NTASK}" -gt 1 ]; then
  python scripts/make_tables.py --runs "${ROOT}" --paper "${ROOT}/paper" || true
else
  python scripts/make_overlap_report.py --runs "${ROOT}" --paper "${ROOT}/paper" || true
fi
python scripts/status.py --runs "${EXP_DIR}" || true
