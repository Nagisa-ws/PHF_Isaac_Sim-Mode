"""Animated USD of a Mission 3 recording for NVIDIA Isaac Sim (the Isaac counterpart of the Meshcat HTML).

MuJoCo ran the evaluation (scripts/eval_xgh.py) and saved recording.npz; this script replays that recording as USD time
samples. No simulation runs here and none runs in Isaac Sim: the USD is visual only, so Isaac Sim shows exactly the
recorded MuJoCo run. Needs `pip install usd-core` (no Isaac Sim, no GPU).

  python scripts/export_usd.py --tag potek_baris_penuh                 # outputs/eval_xgh/<tag>/usd/mission3.usdc
  python scripts/export_usd.py --tag potek_baris_penuh --parts 2       # mission3_part01.usdc, mission3_part02.usdc
  python scripts/export_usd.py --recording some/recording.npz --out some/dir/mission3.usdc --plants-per-row 4

Open the .usdc in Isaac Sim (File > Open), press Play; cameras: viewport camera menu (cam_follow, cam_corner,
cam_wrist, overview) or orbit freely in the Perspective view. Video: scripts/isaac_render_video.py.
The scene is rebuilt with the options stored in the recording (older recordings: pass the options of the run).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pinesim.config import PROJECT_ROOT
from pinesim.envs.xgh_env import XghEvalEnv
from pinesim.viz.usd_export import export_usd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="evaluation folder under --out-root (recording.npz inside it)")
    ap.add_argument("--out-root", default="outputs/eval_xgh")
    ap.add_argument("--recording", default=None, help="recording .npz (instead of --tag)")
    ap.add_argument("--out", default=None, help="output .usdc / .usda (default: <run folder>/usd/mission3.usdc)")
    ap.add_argument("--parts", type=int, default=1, help="split the run into this many USD files of equal duration")
    ap.add_argument("--frames", type=int, nargs=2, default=None, metavar=("F0", "F1"), help="recorded frames [F0, F1)")
    ap.add_argument("--no-scenery", action="store_true", help="without soil disc / rocks / trees / hut / panorama")
    ap.add_argument("--sky", choices=["light", "mesh", "none"], default="light",
                    help="light: panorama photo as dome light (default); mesh: the Meshcat dome mesh; none: plain sky light")
    ap.add_argument("--plants-per-row", type=int, default=None, help="only for recordings without stored options")
    ap.add_argument("--section", type=int, nargs=2, default=None)
    ap.add_argument("--field-seed", type=int, default=None)
    ap.add_argument("--side-beds", type=int, default=None)
    ap.add_argument("--edge-gap", type=float, default=None)
    a = ap.parse_args()
    if not (a.tag or a.recording):
        ap.error("--tag or --recording is required")
    run = (PROJECT_ROOT / a.out_root / a.tag) if a.tag else Path(a.recording).resolve().parent
    rec = Path(a.recording) if a.recording else run / "recording.npz"
    res = json.loads(str(np.load(rec, allow_pickle=False)["result"][0]))
    over = dict(res.get("overrides") or {})
    if a.plants_per_row:
        over.setdefault("eval_field", {})["plants_per_row"] = int(a.plants_per_row)
    if a.section:
        over.setdefault("eval_field", {})["section"] = [int(a.section[0]), int(a.section[1])]
    if a.field_seed is not None:
        over.setdefault("eval_field", {})["seed"] = int(a.field_seed)
    elif res.get("field_seed") is not None:
        over.setdefault("eval_field", {}).setdefault("seed", int(res["field_seed"]))
    if a.edge_gap is not None:
        over.setdefault("field", {})["side_rows"] = {"enabled": a.edge_gap > 0, "edge_gap": float(a.edge_gap)}
    if a.side_beds is not None:
        over.setdefault("field", {}).setdefault("side_rows", {}).update(
            {"enabled": a.side_beds > 0, "beds_per_side": max(1, int(a.side_beds))})
    env = XghEvalEnv("mission3", seed=int(res.get("seed", 0) or 0), overrides=over or None)
    m = env.sim.m
    if res.get("model_size") and list(res["model_size"]) != [int(m.nbody), int(m.ngeom)]:
        sys.exit(f"the rebuilt scene ({m.nbody} bodies, {m.ngeom} geoms) is not the recorded one {res['model_size']}")
    scenery = None if a.no_scenery else dict(env.cfg.get("scenery") or {})
    out = Path(a.out) if a.out else run / "usd" / "mission3.usdc"
    n = len(np.load(rec, allow_pickle=False)["frames"])
    f0, f1 = a.frames if a.frames else (0, n)
    info = dict(title="Mission 3", fruits=f"{res.get('n_placed', '-')} / {res.get('n_targets', '-')}",
                detach=res.get("detach_mode", "twist"))
    cuts = np.linspace(f0, f1, max(1, a.parts) + 1).astype(int)
    for j in range(len(cuts) - 1):
        o = out if a.parts <= 1 else out.with_name(f"{out.stem}_part{j + 1:02d}{out.suffix}")
        r = export_usd(m, rec, o, (int(cuts[j]), int(cuts[j + 1])), scenery=scenery, sky=a.sky, info=info)
        print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
