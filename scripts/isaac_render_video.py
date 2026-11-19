"""Video of a Mission 3 USD replay rendered by NVIDIA Isaac Sim: FOUR CAMERAS IN ONE FRAME (2 x 2).

  follow camera   | corner camera
  wrist camera    | overview camera          + phase, time, fruits in the crates, T1/T2 decision text

Run it with Isaac Sim's Python (it starts Isaac Sim headless, RTX renderer; needs an RTX GPU):

  ./python.sh /path/to/scripts/isaac_render_video.py --usd outputs/eval_xgh/<tag>/usd/mission3.usdc
  ./python.sh .../isaac_render_video.py --usd .../mission3.usdc --start 60 --duration 30          # 30 s from 1:00
  ./python.sh .../isaac_render_video.py --usd .../mission3.usdc --speed 4 --tile 640 360         # 4x time lapse
  ./python.sh .../isaac_render_video.py --usd .../mission3.usdc --cameras follow                 # one camera, full frame
  (pip install of Isaac Sim / Isaac Lab:  python scripts/isaac_render_video.py ...)

The USD comes from scripts/export_usd.py (MuJoCo recording replayed as USD time samples, no physics). The side-car
file <name>_meta.json next to the USD gives the cameras, the phase labels and the fruit counter.
Output: <usd folder>/<name>_4cam.mp4 (OpenCV or imageio; without either: JPEG frames + the ffmpeg command).

  --dry-run   no Isaac Sim: coloured test tiles through the same compositing / video code (checks the installation)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TILE_TITLES = {"follow": "Follow Vehicle", "corner": "Corner Camera", "wrist": "Wrist Camera", "overview": "Overview"}
ORDER = ["follow", "corner", "wrist", "overview"]


# ----------------------------------------------------------------------------- compositing (no Isaac Sim needed)
def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _text(draw, xy, text, font, fill=(255, 255, 255)):
    x, y = xy
    for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


class Hud:
    """Per-frame text of the video from the side-car file: phase label, fruit counter, decision text."""

    def __init__(self, meta: dict):
        self.fps = float(meta["fps"])
        self.labels = meta.get("labels", [])
        self.overlays = meta.get("overlays", [])
        self.placed = meta.get("placed_frames", [])
        self.placed0 = int(meta.get("placed_before", 0))
        self.total = (meta.get("result") or {}).get("n_targets")
        self.title = (meta.get("info") or {}).get("title", "Mission 3")
        self._li = 0

    def label(self, k: int) -> str:
        L = self.labels
        while self._li < len(L) - 1 and k >= L[self._li][1]:
            self._li += 1
        while self._li > 0 and k < L[self._li][0]:
            self._li -= 1
        return L[self._li][2] if L else ""

    def fruits(self, k: int) -> int:
        return self.placed0 + int(np.searchsorted(self.placed, k, side="right"))

    def decision(self, k: int) -> str:
        for o in self.overlays:
            if o["f0"] <= k < o["f1"]:
                return o["text"]
        return ""


def compose(tiles: dict, keys: list, hud: Hud, k: int, tile_wh) -> np.ndarray:
    """2 x 2 frame (or one full tile) with titles and the HUD. tiles: key -> HxWx3 uint8."""
    from PIL import Image, ImageDraw
    W, H = tile_wh
    cols = 1 if len(keys) == 1 else 2
    rows = int(np.ceil(len(keys) / cols))
    im = Image.new("RGB", (cols * W, rows * H), (12, 14, 16))
    dr = ImageDraw.Draw(im)
    f_title, f_small = _font(max(12, H // 24)), _font(max(10, H // 34))
    for i, key in enumerate(keys):
        x, y = (i % cols) * W, (i // cols) * H
        t = tiles.get(key)
        if t is not None:
            im.paste(Image.fromarray(np.ascontiguousarray(t[:, :, :3])).resize((W, H)) if t.shape[:2] != (H, W)
                     else Image.fromarray(np.ascontiguousarray(t[:, :, :3])), (x, y))
        _text(dr, (x + 10, y + 8), TILE_TITLES.get(key, key), f_title)
        if key == "wrist":
            for j, line in enumerate(hud.decision(k).split("\n")):
                _text(dr, (x + 10, y + 12 + f_title.size + j * (f_small.size + 3)), line, f_small, (230, 240, 255))
    if cols > 1:
        dr.line([(W, 0), (W, rows * H)], fill=(255, 255, 255), width=2)
        dr.line([(0, H), (cols * W, H)], fill=(255, 255, 255), width=2)
    t = k / hud.fps
    info = f"{hud.title}   t = {int(t // 60):02d}:{t % 60:04.1f}   phase: {hud.label(k) or '-'}   fruits in crates: {hud.fruits(k)}" \
           + (f" / {hud.total}" if hud.total else "")
    f_hud = _font(max(13, H // 22))
    tw = dr.textlength(info, font=f_hud)
    bx, by = (cols * W - tw) / 2, rows * H - f_hud.size - 14
    dr.rectangle([bx - 12, by - 6, bx + tw + 12, by + f_hud.size + 8], fill=(20, 24, 28))
    dr.text((bx, by), info, font=f_hud, fill=(255, 255, 255))
    return np.asarray(im)


class VideoOut:
    """MP4 through OpenCV, else imageio; else JPEG frames (and the ffmpeg command to join them)."""

    def __init__(self, path: Path, fps: float):
        self.path, self.fps, self.n = Path(path), float(fps), 0
        self.cv = self.io = None
        self.frames_dir = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rgb: np.ndarray):
        if self.n == 0:
            h, w = rgb.shape[:2]
            try:
                import cv2
                self.cv2 = cv2
                self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                if not self.cv.isOpened():
                    self.cv = None
            except Exception:
                self.cv = None
            if self.cv is None:
                try:
                    import imageio
                    self.io = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", quality=8, macro_block_size=2)
                except Exception:
                    self.io = None
            if self.cv is None and self.io is None:
                self.frames_dir = self.path.with_suffix("")
                self.frames_dir.mkdir(parents=True, exist_ok=True)
        if self.cv is not None:
            self.cv.write(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))
        elif self.io is not None:
            self.io.append_data(rgb)
        else:
            from PIL import Image
            Image.fromarray(rgb).save(self.frames_dir / f"frame_{self.n:06d}.jpg", quality=92)
        self.n += 1

    def close(self) -> str:
        if self.cv is not None:
            self.cv.release()
            return str(self.path)
        if self.io is not None:
            self.io.close()
            return str(self.path)
        return (f"{self.frames_dir}/frame_%06d.jpg  (no OpenCV / imageio in this Python; join the frames with:\n"
                f"  ffmpeg -framerate {self.fps:g} -i {self.frames_dir}/frame_%06d.jpg -c:v libx264 -pix_fmt yuv420p {self.path})")


# ----------------------------------------------------------------------------- renderers
class DryRenderer:
    """Test tiles instead of Isaac Sim (--dry-run)."""

    def __init__(self, keys, tile_wh):
        self.keys, (self.W, self.H) = keys, tile_wh

    def render(self, k: int, time_s: float) -> dict:
        out = {}
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        for i, key in enumerate(self.keys):
            t = np.zeros((self.H, self.W, 3), np.uint8)
            t[..., i % 3] = 60 + (xx + 3 * k) % 120
            t[..., (i + 1) % 3] = 40 + (yy * 100 // self.H)
            out[key] = t
        return out

    def close(self):
        pass


class IsaacRenderer:
    """Isaac Sim headless: one render product per camera; the timeline time is set, then one Replicator step renders
    all cameras of that time (synchronous: the pictures belong to the time that was set)."""

    def __init__(self, usd: Path, cam_paths: dict, keys, tile_wh, subframes: int, renderer: str | None, warmup: int):
        from isaacsim import SimulationApp
        cfg = {"headless": True, "width": tile_wh[0], "height": tile_wh[1], "open_usd": str(usd)}
        if renderer:
            cfg["renderer"] = renderer
        self.app = SimulationApp(cfg)
        import carb
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        self.rep = rep
        ctx = omni.usd.get_context()
        for _ in range(10):
            self.app.update()
        stage = ctx.get_stage()
        if stage is None or not stage.GetPrimAtPath("/World").IsValid():      # open_usd not honoured: open it here
            if not ctx.open_stage(str(usd)):
                raise RuntimeError(f"cannot open {usd}")
            for _ in range(10):
                self.app.update()
        while ctx.get_stage_loading_status()[2] > 0:           # files still loading
            self.app.update()
        stage = ctx.get_stage()
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)   # frames are captured by step() only
        try:
            rep.orchestrator.set_capture_on_play(False)
        except Exception:
            pass
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.stop()
        self.subframes = int(subframes)
        self.rps, self.ann = {}, {}
        for key in keys:
            path = cam_paths[key]
            if not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"camera {path} is not in the stage")
            rp = rep.create.render_product(path, tile_wh, name=f"rp_{key}")
            try:
                an = rep.annotators.get("rgb")
            except Exception:                                   # older Replicator
                an = rep.AnnotatorRegistry.get_annotator("rgb")
            an.attach(rp)
            self.rps[key], self.ann[key] = rp, an
        self.timeline.set_current_time(0.0)
        for _ in range(max(1, warmup)):                         # shaders, textures, first frames
            self._step(self.subframes)

    def _step(self, n: int):
        try:
            self.rep.orchestrator.step(rt_subframes=n, delta_time=0.0, pause_timeline=True)
        except TypeError:                                       # older Replicator: no delta_time
            self.rep.orchestrator.step(rt_subframes=n, pause_timeline=True)

    def render(self, k: int, time_s: float) -> dict:
        self.timeline.set_current_time(float(time_s))
        self._step(self.subframes)
        out = {}
        for key, an in self.ann.items():
            d = an.get_data()
            d = d["data"] if isinstance(d, dict) else d
            out[key] = np.asarray(d)[:, :, :3].astype(np.uint8)
        return out

    def close(self):
        for an in self.ann.values():
            an.detach()
        for rp in self.rps.values():
            rp.destroy()
        self.app.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", required=True, help="mission3.usdc of scripts/export_usd.py (its _meta.json beside it)")
    ap.add_argument("--out", default=None, help="output .mp4 (default: <usd name>_4cam.mp4 beside the USD)")
    ap.add_argument("--cameras", nargs="+", default=ORDER, choices=ORDER, help="tiles, in order (default: all four)")
    ap.add_argument("--tile", type=int, nargs=2, default=[960, 540], metavar=("W", "H"), help="size of one camera tile")
    ap.add_argument("--start", type=float, default=0.0, help="first second of the replay")
    ap.add_argument("--duration", type=float, default=None, help="seconds of the replay (default: to the end)")
    ap.add_argument("--speed", type=float, default=1.0, help="time lapse: only every `speed`-th frame is rendered")
    ap.add_argument("--fps", type=float, default=None, help="frame rate of the video (default: the USD rate)")
    ap.add_argument("--subframes", type=int, default=4, help="RTX sub-frames per picture (more = cleaner, slower)")
    ap.add_argument("--renderer", default=None, help="Isaac Sim rendering mode: RealTimePathTracing | RaytracedLighting | "
                    "PathTracing (default: the default of the installed Isaac Sim)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true", help="test tiles instead of Isaac Sim")
    a = ap.parse_args()
    usd = Path(a.usd).resolve()
    meta_file = usd.with_name(usd.stem + "_meta.json")
    if not usd.exists() or not meta_file.exists():
        sys.exit(f"need {usd} and {meta_file} (python scripts/export_usd.py ...)")
    meta = json.loads(meta_file.read_text())
    keys = [k for k in a.cameras if k in meta["cameras"]]
    if not keys:
        sys.exit(f"none of the cameras {a.cameras} is in the USD (it has {list(meta['cameras'])})")
    fps, n = float(meta["fps"]), int(meta["frames"])
    k0 = min(n - 1, max(0, int(round(a.start * fps))))
    k1 = n if a.duration is None else min(n, k0 + max(1, int(round(a.duration * fps))))
    frames = np.unique(np.round(np.arange(k0, k1, max(a.speed, 1e-6))).astype(int))
    frames = frames[frames < k1]
    out = Path(a.out) if a.out else usd.with_name(f"{usd.stem}_{len(keys)}cam.mp4")
    tile = (int(a.tile[0]), int(a.tile[1]))
    ren = DryRenderer(keys, tile) if a.dry_run else IsaacRenderer(usd, meta["cameras"], keys, tile, a.subframes,
                                                                a.renderer, a.warmup)
    hud, vid = Hud(meta), VideoOut(out, a.fps or fps)
    print(f"{len(frames)} frames ({frames[0] / fps:.1f} s .. {frames[-1] / fps:.1f} s of the replay), cameras {keys}, "
          f"{len(keys) if len(keys) == 1 else 2 * tile[0]} px wide -> {out}", flush=True)
    try:
        for i, k in enumerate(frames):
            vid.write(compose(ren.render(int(k), k / fps), keys, hud, int(k), tile))
            if i % 100 == 0:
                print(f"  frame {i + 1} / {len(frames)}", flush=True)
    finally:
        print("->", vid.close(), flush=True)
        ren.close()


if __name__ == "__main__":
    main()
