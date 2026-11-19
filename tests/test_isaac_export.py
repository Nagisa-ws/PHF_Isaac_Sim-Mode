"""Isaac Sim replay (USD export, video compositing) and the "potek" detach variant.

  pytest -q tests/test_isaac_export.py          (needs usd-core; no Isaac Sim, no GPU)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def env():
    from pinesim.envs.xgh_env import XghEvalEnv
    e = XghEvalEnv("mission3", seed=1, overrides={"eval_field": {"plants_per_row": 2},
                                                  "realistic": {"stem_break_bend_range": [8.0, 14.0]},
                                                  "gripper": {"pad_friction": 1.8}})
    e._scene_reset()
    return e


def test_twist_is_still_the_default():
    from pinesim.config import load_config
    cfg = load_config("mission3")
    assert cfg["harvest"]["detach_mode"] == "twist"
    assert cfg["realistic"].get("stem_break_bend_range") is None        # bending never snaps a stem unless asked for
    assert cfg["gripper"].get("pad_friction") is None


def test_pad_friction_override(env):
    sim = env.sim
    pads = sorted(sim.pad_left | sim.pad_right)
    assert pads and np.allclose(sim.m.geom_friction[pads, 0], 1.8)


def test_stem_snaps_in_bending_plus_tension(env):
    """N / N_break + M / M_break >= 1; the bending strength has the rank of the torsion strength in its range."""
    sim, lod = env.sim, env.lod
    lod.set_interactive(lod.wanted(0, float(sim.base_pos[0])))
    i = lod.state[0]["slot"]
    fld, d = sim.field, sim.d
    s, lay = fld.slots[i], fld.slots[i]["layout"]
    assert 8.0 <= lay.break_bend <= 14.0
    t0, t1 = env.cfg["realistic"]["stem_break_torque_range"]
    assert lay.break_bend == pytest.approx(8.0 + (lay.break_torque - t0) / (t1 - t0) * 6.0)
    fld.record_rest_load(i, d)
    q0 = d.qpos.copy()
    bend_at = lambda frac: frac * lay.break_bend / s["k_bend"]
    d.qpos[s["stem_qadr"][1]] = s["rest_bend"][0] + bend_at(0.9)
    assert not fld.check_break(i, d)[0]                                  # 90 % of the bending strength alone: holds
    d.qpos[s["stem_qadr"][3]] = s["rest_ext"] + 0.2 * lay.break_force / s["k_pull"]
    assert fld.check_break(i, d)[0]                                      # + 20 % of the tensile strength: snaps
    d.qpos[s["stem_qadr"][1]] = s["rest_bend"][0] + bend_at(1.05)
    d.qpos[s["stem_qadr"][3]] = s["rest_ext"]
    assert fld.check_break(i, d)[0]                                      # pure bending beyond the strength: snaps
    lay.break_bend = 0.0
    assert not fld.check_break(i, d)[0]                                  # no bending strength set: only bends (as before)
    d.qpos[:] = q0


def test_gltf_is_turned_to_z_up():
    from pinesim.config import PROJECT_ROOT
    from pinesim.viz.usd_export import Gltf
    pr = next(Gltf(PROJECT_ROOT / "assets_xgh" / "plants" / "p1_static.gltf").primitives())
    P = pr["P"]
    assert P[:, 2].min() > -0.05 and 0.5 < P[:, 2].max() < 1.6           # a plant stands on z = 0 and grows upward
    assert pr["C"] is not None and pr["C"].shape == (len(P), 3) and pr["C"].max() <= 1.0


def test_export_usd_matches_the_recording(env, tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    from pinesim.viz.usd_export import export_usd
    from pinesim.viz.xgh_recorder import XghRecorder
    sim, m, d = env.sim, env.sim.m, env.sim.d
    rec = XghRecorder(sim, every=1)
    rec("scan")
    b = m.body("vehicle").id
    for k in range(1, 6):                                                # vehicle "moves" 5 cm per frame from frame 3 on
        if k >= 3:
            d.xpos[b] = d.xpos[b] + [0.05, 0.0, 0.0]
        rec("drive" if k >= 3 else "scan")
    rec.overlay(t1=[1.0, 0.0, 0.8], t2=[1.2, 0.0, 0.7], info=dict(method="diffusion", n_candidates=3))
    final = d.xpos[b].copy()
    rec.save(tmp_path / "rec.npz", dict(seed=1, n_placed=0, n_targets=4, events=["4: fruit in crate basket (1)"]))
    r = export_usd(m, tmp_path / "rec.npz", tmp_path / "usd" / "m3.usdc", scenery={})
    st = Usd.Stage.Open(r["usd"])
    assert UsdGeom.GetStageUpAxis(st) == "Z" and UsdGeom.GetStageMetersPerUnit(st) == 1.0
    assert st.GetTimeCodesPerSecond() == pytest.approx(1.0 / sim.dt) and st.GetEndTimeCode() == r["frames"] - 1
    veh = st.GetPrimAtPath("/World/scene/vehicle")
    tr = veh.GetAttribute("xformOp:translate")
    assert np.allclose(tr.Get(r["frames"] - 1), final, atol=1e-9)
    # held until the frame before the first move (hold key), not drifting toward it
    assert np.allclose(tr.Get(1.5), tr.Get(0), atol=1e-12) and not np.allclose(tr.Get(3), tr.Get(0))
    for key in ("follow", "corner", "wrist", "overview"):
        assert st.GetPrimAtPath(json.loads((tmp_path / "usd" / "m3_meta.json").read_text())["cameras"][key]).IsA(UsdGeom.Camera)
    assert st.GetPrimAtPath("/World/scenery/hut").IsValid() and st.GetPrimAtPath("/World/lights/sky").IsValid()
    assert st.GetPrimAtPath("/World/overlay/d000/T1").IsValid()
    assert (tmp_path / "usd" / "textures" / "panorama.jpg").exists()
    meta = json.loads((tmp_path / "usd" / "m3_meta.json").read_text())
    assert meta["placed_frames"] == [4] and meta["labels"][0][2] == "scan"
    # mesh geoms sit where MuJoCo has them: world box of the USD instance = box of MuJoCo's compiled mesh (frame 0)
    from pinesim.viz.xgh_recorder import visual_geoms
    pose0 = {int(x[1]): (np.array(x[2:5]), np.array(x[5:9])) for x in rec.bchg if x[0] == 0}
    name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
    picks = [g for g in visual_geoms(m) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
             and "_leaves_" not in name(mujoco.mjtObj.mjOBJ_GEOM, g) and "_fill_" not in name(mujoco.mjtObj.mjOBJ_GEOM, g)
             and pose0.get(int(m.geom_bodyid[g]), (np.zeros(3),))[0][2] > -1.0][::17]
    assert len(picks) > 10
    cache = UsdGeom.XformCache(Usd.TimeCode(0))

    def rot(q):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(q, float))
        return R.reshape(3, 3)

    for g in picks:
        bid = int(m.geom_bodyid[g])
        pb, qb = pose0.get(bid, (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
        prim = st.GetPrimAtPath(f"/World/scene/{name(mujoco.mjtObj.mjOBJ_BODY, bid)}/g{g}")
        assert prim.IsValid()
        pts = []
        for c in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if c.IsA(UsdGeom.Mesh):
                M = np.array(cache.GetLocalToWorldTransform(c))
                pts.append(np.array(UsdGeom.Mesh(c).GetPointsAttr().Get()) @ M[:3, :3] + M[3, :3])
        U = np.concatenate(pts)
        did = int(m.geom_dataid[g])
        Rw = rot(qb) @ rot(m.geom_quat[g])
        V = m.mesh_vert[m.mesh_vertadr[did]:m.mesh_vertadr[did] + m.mesh_vertnum[did]] @ Rw.T + pb + rot(qb) @ m.geom_pos[g]
        assert np.abs(U.min(0) - V.min(0)).max() < 3e-3 and np.abs(U.max(0) - V.max(0)).max() < 3e-3, name(mujoco.mjtObj.mjOBJ_GEOM, g)


def test_video_compositing_without_isaac(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import isaac_render_video as V
    meta = dict(fps=25.0, frames=100, cameras={k: f"/World/{k}" for k in V.ORDER}, labels=[[0, 50, "look"], [50, 100, "potek"]],
                overlays=[dict(f0=10, f1=60, text="T1/T2")], placed_frames=[20, 70], placed_before=3, result=dict(n_targets=60))
    hud = V.Hud(meta)
    assert (hud.label(10), hud.label(60), hud.label(49)) == ("look", "potek", "look")
    assert (hud.fruits(0), hud.fruits(20), hud.fruits(99)) == (3, 4, 5)
    assert hud.decision(5) == "" and hud.decision(30) == "T1/T2"
    ren = V.DryRenderer(V.ORDER, (160, 90))
    frame = V.compose(ren.render(7, 0.28), V.ORDER, hud, 7, (160, 90))
    assert frame.shape == (180, 320, 3) and frame.dtype == np.uint8
    one = V.compose(ren.render(7, 0.28), ["wrist"], hud, 7, (160, 90))
    assert one.shape == (90, 160, 3)
