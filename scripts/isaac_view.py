"""Open a Mission 3 USD replay in the NVIDIA Isaac Sim window and play it (free camera: orbit with Alt + mouse).

  ./python.sh /path/to/scripts/isaac_view.py --usd outputs/eval_xgh/<tag>/usd/mission3.usdc [--camera follow] [--start 60]

The same as File > Open in Isaac Sim followed by Play; --camera picks the viewport camera (follow | corner | wrist |
overview; default: Isaac Sim's free Perspective camera). The viewport's camera menu switches between them at any time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--usd", required=True)
ap.add_argument("--camera", default=None, choices=["follow", "corner", "wrist", "overview"])
ap.add_argument("--start", type=float, default=0.0, help="second of the replay to start at")
ap.add_argument("--no-loop", action="store_true")
a = ap.parse_args()
usd = Path(a.usd).resolve()

from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "open_usd": str(usd)})

import omni.timeline
import omni.usd

ctx = omni.usd.get_context()
for _ in range(10):
    app.update()
if ctx.get_stage() is None or not ctx.get_stage().GetPrimAtPath("/World").IsValid():   # open_usd not honoured
    if not ctx.open_stage(str(usd)):
        raise SystemExit(f"cannot open {usd}")
    for _ in range(10):
        app.update()
while ctx.get_stage_loading_status()[2] > 0:
    app.update()
meta_file = usd.with_name(usd.stem + "_meta.json")
if a.camera and meta_file.exists():
    path = json.loads(meta_file.read_text())["cameras"].get(a.camera)
    if path:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = path
tl = omni.timeline.get_timeline_interface()
tl.set_looping(not a.no_loop)
tl.set_current_time(float(a.start))
tl.play()
while app.is_running():
    app.update()
app.close()
