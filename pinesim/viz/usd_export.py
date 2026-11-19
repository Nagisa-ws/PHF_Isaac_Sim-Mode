"""
pinesim/viz/usd_export.py
-------------------------
Recording (.npz of XghRecorder) -> ANIMATED USD for NVIDIA Isaac Sim (the Isaac counterpart of the Meshcat HTML).

MuJoCo simulates, XghRecorder stores the body poses, and this module replays them as USD time samples - exactly the
role Meshcat has in export_html(): no physics in the USD, so what Isaac Sim shows is the recorded MuJoCo run.

  export_usd(model, recording, out_dir)   ->  <out_dir>/<name>.usdc + textures/ + <name>_meta.json

Stage layout (Z up, metres, timeCodesPerSecond = recording rate):
  /World/scene/<body>            one Xform per MuJoCo body, world pose as time samples (translate + orient)
  /World/scene/<body>/g<id>      its visual geoms: instanced mesh (assets_xgh glTF) or primitive, visibility animated
                                 (crate filling up, picked fruits, the indicator lamp, LOD swaps)
  /World/scene/<body>/cam_<key>  the scene's cameras: follow, corner, wrist  (+ /World/cameras/overview)
  /World/overlay, overlay_wrist  T1 / T2 decision of every target (visible while it is current)
  /World/scenery                 the decoration of the Meshcat HTML: soil, grass, rocks, trees, stumps, hut
  /World/lights                  sky (dome light with the panorama photo) + sun
  /_Prototypes                   every mesh once (class prims, referenced as instances)

Mesh frames: the glTF files are +Y up (Drake turns them back when it loads them) and keep the ORIGINAL mesh
coordinates, while MuJoCo recentres meshes - so the points are turned to +Z up here and every mesh geom sits at
T_geom_compiled * inverse(T_mesh) (geom_local_original, the same rule as the Meshcat export).

Needs only `pip install usd-core` (no Isaac Sim, no GPU); Isaac Sim is needed to LOOK at the result.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import mujoco
import numpy as np

from pinesim.config import PROJECT_ROOT
from pinesim.viz.xgh_recorder import camera_nodes, geom_local_original, mesh_gltf, overlay_text, visual_geoms

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
CAMERA_ASPECT = 16.0 / 9.0
H_APERTURE = 20.955                      # mm (USD default); vertical aperture follows the aspect
PARK_Z = -2.0                            # bodies below this are parked LOD slots: hidden


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _ident(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return s if re.match(r"[A-Za-z_]", s) else "_" + s


# ----------------------------------------------------------------------------- glTF (the subset the assets use)
class Gltf:
    """Triangle meshes of a glTF 2.0 file: positions (turned to +Z up), vertex colours, UVs, base-colour material."""

    def __init__(self, path):
        self.path = Path(path)
        self.j = json.loads(self.path.read_text())
        self.buf = []
        for b in self.j.get("buffers", []):
            uri = b["uri"]
            self.buf.append(base64.b64decode(uri.split(",", 1)[1]) if uri.startswith("data:")
                            else (self.path.parent / uri).read_bytes())
        for n in self.j.get("nodes", []):
            if any(k in n for k in ("matrix", "translation", "rotation", "scale", "children")):
                raise NotImplementedError(f"{self.path.name}: node transforms are not supported")

    def acc(self, i: int) -> np.ndarray:
        a = self.j["accessors"][i]
        if "sparse" in a:
            raise NotImplementedError("sparse accessor")
        bv = self.j["bufferViews"][a["bufferView"]]
        dt, n = np.dtype(_CT[a["componentType"]]), _NC[a["type"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride") or dt.itemsize * n
        raw = self.buf[bv["buffer"]]
        if stride == dt.itemsize * n:
            return np.frombuffer(raw, dt, a["count"] * n, off).reshape(a["count"], n)
        return np.array(np.ndarray((a["count"], n), dt, raw, off, (stride, dt.itemsize)))

    def image(self, i: int) -> tuple[bytes, str]:
        im = self.j["images"][i]
        if "uri" in im and not im["uri"].startswith("data:"):
            f = self.path.parent / im["uri"]
            return f.read_bytes(), f.suffix.lower()
        if "uri" in im:
            head, data = im["uri"].split(",", 1)
            return base64.b64decode(data), (".png" if "png" in head else ".jpg")
        bv = self.j["bufferViews"][im["bufferView"]]
        o = bv.get("byteOffset", 0)
        return self.buf[bv["buffer"]][o:o + bv["byteLength"]], (".png" if "png" in im.get("mimeType", "") else ".jpg")

    def primitives(self):
        for mesh in self.j["meshes"]:
            for pr in mesh["primitives"]:
                if pr.get("mode", 4) != 4:
                    continue
                at = pr["attributes"]
                P = self.acc(at["POSITION"]).astype(np.float32)
                P = np.stack([P[:, 0], -P[:, 2], P[:, 1]], axis=1)                    # glTF +Y up -> +Z up
                F = (self.acc(pr["indices"]).reshape(-1, 3) if "indices" in pr
                     else np.arange(len(P)).reshape(-1, 3)).astype(np.int32)
                C = A = None
                if "COLOR_0" in at:
                    c = self.acc(at["COLOR_0"])
                    c = c.astype(np.float32) / (np.iinfo(c.dtype).max if c.dtype.kind in "ui" else 1.0)
                    C, A = c[:, :3], (c[:, 3] if c.shape[1] > 3 else None)
                UV = None
                if "TEXCOORD_0" in at:
                    uv = self.acc(at["TEXCOORD_0"]).astype(np.float32)
                    UV = np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1)                 # glTF v runs down, USD t up
                mat = self.j["materials"][pr["material"]] if "material" in pr else None
                yield dict(P=P, F=F, C=C, A=A, UV=UV, mat=mat)


# ----------------------------------------------------------------------------- USD writer
class UsdScene:
    def __init__(self, out_file: Path, fps: float, n_frames: int):
        from pxr import Sdf, Usd, UsdGeom
        self.Sdf, self.Usd, self.UsdGeom = Sdf, Usd, UsdGeom
        self.out = Path(out_file)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.tex_dir = self.out.parent / "textures"
        if self.out.exists():
            self.out.unlink()
        self.stage = st = Usd.Stage.CreateNew(str(self.out))
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        st.SetTimeCodesPerSecond(fps)
        st.SetFramesPerSecond(fps)
        st.SetStartTimeCode(0)
        st.SetEndTimeCode(max(0, n_frames - 1))
        self.world = UsdGeom.Xform.Define(st, "/World")
        st.SetDefaultPrim(self.world.GetPrim())
        self.layer = st.GetRootLayer()
        st.CreateClassPrim("/_Prototypes")
        self.protos = {}
        self.textures = {}
        self.n_tris = 0

    # -- transforms
    def xform(self, path: str, pos=None, quat=None, scale=None):
        from pxr import Gf
        x = self.UsdGeom.Xform.Define(self.stage, path)
        self.set_ops(x, pos, quat, scale)
        return x

    def set_ops(self, xf, pos=None, quat=None, scale=None):
        from pxr import Gf
        if pos is not None:
            xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
        if quat is not None:
            q = np.asarray(quat, float)
            q = q / np.linalg.norm(q)
            xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        if scale is not None:
            s = np.broadcast_to(np.asarray(scale, float), (3,))
            xf.AddScaleOp().Set(Gf.Vec3f(float(s[0]), float(s[1]), float(s[2])))

    def pose_samples(self, path: str, frames, P, Q):
        """Time samples of xformOp:translate / orient with a HOLD key before every jump (USD interpolates linearly
        between samples: a pose that is held and then changes would otherwise drift slowly toward the next one - the
        same rule as the Meshcat export)."""
        from pxr import Gf
        tp, op = self.Sdf.Path(f"{path}.xformOp:translate"), self.Sdf.Path(f"{path}.xformOp:orient")
        lay = self.layer
        last_k, last_p, last_q = None, None, None
        for k, p, q in zip(frames, P, Q):
            k = int(k)
            if last_q is not None and float(np.dot(q, last_q)) < 0.0:
                q = -q                                                   # same hemisphere: slerp takes the short way
            vp, vq = Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])), Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            if last_k is not None and k - last_k > 1:
                lay.SetTimeSample(tp, k - 1, last_p)
                lay.SetTimeSample(op, k - 1, last_q_v)
            lay.SetTimeSample(tp, k, vp)
            lay.SetTimeSample(op, k, vq)
            last_k, last_p, last_q, last_q_v = k, vp, q, vq

    def visibility_samples(self, path: str, frames, visible):
        vp = self.Sdf.Path(f"{path}.visibility")
        last = None
        for k, v in zip(frames, visible):
            if last is None or bool(v) != last:
                self.layer.SetTimeSample(vp, int(k), "inherited" if v else "invisible")
                last = bool(v)

    def vis_attr(self, path: str, initial: bool | None = None):
        a = self.UsdGeom.Imageable(self.stage.GetPrimAtPath(path)).CreateVisibilityAttr()
        if initial is not None:
            a.Set("inherited" if initial else "invisible")
        return a

    # -- materials / meshes
    def texture_file(self, data: bytes, name: str) -> str:
        self.tex_dir.mkdir(parents=True, exist_ok=True)
        if name not in self.textures:
            (self.tex_dir / name).write_bytes(data)
            self.textures[name] = f"./textures/{name}"
        return self.textures[name]

    def material(self, path: str, mat: dict, tex: str | None):
        from pxr import Gf, Sdf, UsdShade
        pbr = (mat or {}).get("pbrMetallicRoughness", {})
        f = [float(x) for x in pbr.get("baseColorFactor", [1, 1, 1, 1])]
        unlit = "KHR_materials_unlit" in (mat or {}).get("extensions", {})
        m = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, f"{path}/surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(pbr.get("roughnessFactor", 0.9)))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(pbr.get("metallicFactor", 0.0)))
        col = sh.CreateInput("emissiveColor" if unlit else "diffuseColor", Sdf.ValueTypeNames.Color3f)
        if unlit:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
        if tex:
            rd = UsdShade.Shader.Define(self.stage, f"{path}/st")
            rd.CreateIdAttr("UsdPrimvarReader_float2")
            rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tx = UsdShade.Shader.Define(self.stage, f"{path}/tex")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex))
            tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*f))
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
            col.ConnectToSource(tx.ConnectableAPI(), "rgb")
            mode = (mat or {}).get("alphaMode")
            if mode in ("MASK", "BLEND"):
                sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(tx.ConnectableAPI(), "a")
                # cut-out for both: foliage cards; a soft blend is not portable across renderers
                sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(float((mat or {}).get("alphaCutoff", 0.5)))
        else:
            col.Set(Gf.Vec3f(*f[:3]))
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return m

    def mesh(self, path: str, P, F, C=None, UV=None, color=None, no_shadow: bool = False):
        from pxr import Gf, Sdf, UsdGeom, Vt
        me = UsdGeom.Mesh.Define(self.stage, path)
        me.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P, np.float32)))
        me.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        me.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(F, np.int32).reshape(-1)))
        me.CreateSubdivisionSchemeAttr("none")
        me.CreateDoubleSidedAttr(True)
        me.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*P.min(0).tolist()), Gf.Vec3f(*P.max(0).tolist())]))
        pv = UsdGeom.PrimvarsAPI(me.GetPrim())
        if C is not None:
            pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(
                Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(C, np.float32)))
        elif color is not None:
            me.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color[:3]])])
        if UV is not None:
            pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
                Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(UV, np.float32)))
        if no_shadow:
            pv.CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        self.n_tris += len(F)
        return me

    def prototype(self, gltf_path: Path, fallback_rgb=None, drop_alpha_below: float | None = None,
                  no_shadow: bool = False, lit: bool = False) -> str:
        """Class prim with the meshes of one glTF file (made once); returns its path. Vertex colours become
        displayColor; a textured material becomes a UsdPreviewSurface inside the prototype. `lit`: an unlit glTF
        material is written as an ordinary lit one (only the sky dome has to glow)."""
        from pxr import UsdShade
        k0 = (str(gltf_path), None)
        kc = (str(gltf_path), None if fallback_rgb is None else tuple(np.round(fallback_rgb, 4).tolist()))
        for k in (k0, kc):
            if k in self.protos:
                return self.protos[k]
        rel = gltf_path.relative_to(PROJECT_ROOT / "assets_xgh") if (PROJECT_ROOT / "assets_xgh") in gltf_path.parents else gltf_path
        name = _ident("_".join(rel.with_suffix("").parts))
        if any(p.endswith("/" + name) for p in self.protos.values()):
            name += f"_c{len(self.protos)}"
        root = f"/_Prototypes/{name}"
        self.UsdGeom.Xform.Define(self.stage, root)
        g = Gltf(gltf_path)
        used_fallback = False
        for i, pr in enumerate(g.primitives()):
            P, F, C = pr["P"], pr["F"], pr["C"]
            mat = pr["mat"]
            if lit and mat is not None:
                mat = {k: v for k, v in mat.items() if k != "extensions"}
            ti = (mat or {}).get("pbrMetallicRoughness", {}).get("baseColorTexture", {}).get("index") if pr["UV"] is not None else None
            if drop_alpha_below is not None and pr["A"] is not None:      # vertex-alpha fade (grass ring): keep the solid part
                F = F[pr["A"][F].mean(axis=1) >= drop_alpha_below]
                if not len(F):
                    continue
            if ti is not None:
                data, ext = g.image(g.j["textures"][ti]["source"])
                tex = self.texture_file(data, f"{name}_{ti}{ext}")
                me = self.mesh(f"{root}/m{i}", P, F, UV=pr["UV"], no_shadow=no_shadow)
                m = self.material(f"{root}/Looks/mat{i}", mat, tex)
                UsdShade.MaterialBindingAPI.Apply(me.GetPrim()).Bind(m)
            elif C is not None:
                self.mesh(f"{root}/m{i}", P, F, C=C, no_shadow=no_shadow)
            else:
                f = (mat or {}).get("pbrMetallicRoughness", {}).get("baseColorFactor")
                if f is None:
                    used_fallback = True
                rgb = f[:3] if f is not None else (fallback_rgb if fallback_rgb is not None else [0.6, 0.6, 0.6])
                self.mesh(f"{root}/m{i}", P, F, color=rgb, no_shadow=no_shadow)
        self.protos[kc if used_fallback else k0] = root
        return root

    def instance(self, path: str, proto: str, pos=None, quat=None, scale=None):
        x = self.xform(path, pos, quat, scale)
        x.GetPrim().GetReferences().AddInternalReference(proto)
        x.GetPrim().SetInstanceable(True)
        return x

    def sphere(self, path: str, center, radius: float, rgb):
        from pxr import Gf
        s = self.UsdGeom.Sphere.Define(self.stage, path)
        s.CreateRadiusAttr(float(radius))
        s.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius), Gf.Vec3f(radius, radius, radius)])
        s.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in rgb])])
        s.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        return s

    def polyline(self, path: str, pts, width: float, rgb):
        from pxr import Gf, Vt
        pts = np.asarray(pts, np.float32).reshape(-1, 3)
        c = self.UsdGeom.BasisCurves.Define(self.stage, path)
        c.CreateTypeAttr("linear")
        c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        c.CreateCurveVertexCountsAttr([len(pts)])
        c.CreateWidthsAttr([float(width)])
        c.SetWidthsInterpolation(self.UsdGeom.Tokens.constant)
        c.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in rgb])])
        c.CreateExtentAttr([Gf.Vec3f(*(pts.min(0) - width).tolist()), Gf.Vec3f(*(pts.max(0) + width).tolist())])
        return c

    def camera(self, path: str, fovy_deg: float, pos=None, quat=None, near: float = 0.05, far: float = 500.0):
        """USD camera = MuJoCo camera frame (looks along -Z, +Y up). The vertical field of view is kept; the
        horizontal one follows the 16:9 picture of the video."""
        from pxr import Gf
        cam = self.UsdGeom.Camera.Define(self.stage, path)
        va = H_APERTURE / CAMERA_ASPECT
        cam.CreateHorizontalApertureAttr(H_APERTURE)
        cam.CreateVerticalApertureAttr(va)
        cam.CreateFocalLengthAttr(float(0.5 * va / np.tan(np.radians(fovy_deg) / 2)))
        cam.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))
        self.set_ops(cam, pos, quat)
        return cam


def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """wxyz of a camera at `eye` looking at `target` (camera -Z forward, +Y up)."""
    z = np.asarray(eye, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).reshape(-1))
    return q


def _geom_rgb(m, g):
    c = m.geom_rgba[g].copy()
    mid = int(m.geom_matid[g])
    if mid >= 0 and np.allclose(c, [0.5, 0.5, 0.5, 1.0]):
        c = m.mat_rgba[mid].copy()
    return srgb_to_linear(c[:3])


def _primitive(sc: UsdScene, path: str, m, g) -> bool:
    """MuJoCo primitive geom -> USD gprim at `path` (axis Z, like MuJoCo)."""
    from pxr import Gf
    UsdGeom = sc.UsdGeom
    t, sz, rgb = int(m.geom_type[g]), m.geom_size[g], _geom_rgb(m, g)
    T = mujoco.mjtGeom
    if t == T.mjGEOM_BOX:
        p = UsdGeom.Cube.Define(sc.stage, path)
        p.CreateSizeAttr(1.0)
        p.AddScaleOp().Set(Gf.Vec3f(float(2 * sz[0]), float(2 * sz[1]), float(2 * sz[2])))
        p.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    elif t == T.mjGEOM_SPHERE:
        p = UsdGeom.Sphere.Define(sc.stage, path)
        p.CreateRadiusAttr(float(sz[0]))
        p.CreateExtentAttr([Gf.Vec3f(*[-float(sz[0])] * 3), Gf.Vec3f(*[float(sz[0])] * 3)])
    elif t == T.mjGEOM_ELLIPSOID:
        p = UsdGeom.Sphere.Define(sc.stage, path)
        p.CreateRadiusAttr(1.0)
        p.AddScaleOp().Set(Gf.Vec3f(float(sz[0]), float(sz[1]), float(sz[2])))
        p.CreateExtentAttr([Gf.Vec3f(-1, -1, -1), Gf.Vec3f(1, 1, 1)])
    elif t in (T.mjGEOM_CAPSULE, T.mjGEOM_CYLINDER):
        r, h = max(float(sz[0]), 1e-4), max(float(2 * sz[1]), 1e-4)
        p = (UsdGeom.Capsule if t == T.mjGEOM_CAPSULE else UsdGeom.Cylinder).Define(sc.stage, path)
        p.CreateRadiusAttr(r)
        p.CreateHeightAttr(h)
        p.CreateAxisAttr("Z")
        e = h / 2 + (r if t == T.mjGEOM_CAPSULE else 0.0)
        p.CreateExtentAttr([Gf.Vec3f(-r, -r, -e), Gf.Vec3f(r, r, e)])
    else:
        return False
    p.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in rgb])])
    return True


def _dome_azimuth() -> float | None:
    """World azimuth (rad) of the panorama column u = 0 on the Meshcat dome, and the sense of u; None if no dome."""
    f = PROJECT_ROOT / "assets_xgh" / "scenery" / "dome.gltf"
    if not f.exists():
        return None
    pr = next(Gltf(f).primitives())
    P, u = pr["P"], pr["UV"][:, 0]
    r = np.hypot(P[:, 0], P[:, 1])
    keep = r > 0.5 * r.max()
    az = np.arctan2(P[keep, 1], P[keep, 0])
    # az = az0 + s * 2 pi u  ->  fit s in {-1, +1} and az0 on the circle
    best = None
    for s in (-1.0, 1.0):
        d = az - s * 2 * np.pi * u[keep]
        az0 = np.arctan2(np.sin(d).mean(), np.cos(d).mean())
        err = float(np.abs(np.angle(np.exp(1j * (d - az0)))).mean())
        if best is None or err < best[0]:
            best = (err, s, az0)
    return best[2] if best[1] < 0 else None


def export_usd(model: mujoco.MjModel, rec_path, out_file, frame_range=None, scenery: dict | None = None,
               sky: str = "light", info: dict | None = None) -> dict:
    """Animated USD of the recorded frames [f0, f1) (None = all).

    scenery   None = no decoration; a dict (the `scenery:` block of the config, may be empty) = the decoration of the
              Meshcat HTML (pinesim/viz/scenery.py: same seed -> the same rocks, trees, stumps and hut)
    sky       "light" = the panorama photo as a dome LIGHT (lights the scene and is the background);
              "mesh"  = the Meshcat dome mesh (emissive, casts no shadow) + a plain dome light; "none" = lights only
    """
    from pxr import Gf, Sdf, UsdGeom, UsdLux
    z = np.load(rec_path, allow_pickle=False)
    m = model
    nF = len(z["frames"])
    f0, f1 = (0, nF) if frame_range is None else (max(0, int(frame_range[0])), min(nF, int(frame_range[1])))
    dt = float(z["dt"]) * int(z["every"])
    fps = 1.0 / dt
    out_file = Path(out_file)
    sc = UsdScene(out_file, fps, f1 - f0)
    st, layer = sc.stage, sc.layer
    B, G = z["bchg"], z["gchg"]
    if len(B) and int(B[:, 1].max()) >= m.nbody or len(G) and int(G[:, 1].max()) >= m.ngeom:
        raise ValueError("the recording does not belong to this model (body / geom ids out of range): build the model "
                         "with the options of the evaluation run")
    vg = visual_geoms(m)
    vg_set = set(vg)
    bname = lambda b: _ident(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}")
    cams = camera_nodes(m)
    bodies = sorted({int(m.geom_bodyid[g]) for g in vg} | {int(m.body(n).id) for n, _ in cams.values() if n != "world"} | {0})
    bpath = {b: f"/World/scene/{bname(b)}" for b in bodies}
    UsdGeom.Scope.Define(st, "/World/scene")

    # ---- bodies: world pose time samples
    order = np.lexsort((B[:, 0], B[:, 1]))
    Bs = B[order]
    starts = np.r_[0, np.flatnonzero(np.diff(Bs[:, 1])) + 1, len(Bs)]
    rows_of = {int(Bs[a, 1]): Bs[a:b] for a, b in zip(starts[:-1], starts[1:])}
    n_samples = 0
    for b in bodies:
        sc.xform(bpath[b], [0, 0, 0], [1, 0, 0, 0])
    todo = []
    for b in bodies:
        R = rows_of.get(b)
        if R is None:
            continue
        i0 = int(np.searchsorted(R[:, 0], f0, side="right")) - 1          # state at f0 = last row at or before it
        i1 = int(np.searchsorted(R[:, 0], f1, side="left"))
        R = R[max(i0, 0):i1]
        if not len(R):
            continue
        parked = R[:, 4] < PARK_Z
        if parked.any():
            sc.vis_attr(bpath[b])
        todo.append((b, R, parked))
    with Sdf.ChangeBlock():                                               # pure Sdf writes: fast
        for b, R, parked in todo:
            fr = np.maximum(R[:, 0].astype(int) - f0, 0)
            sc.pose_samples(bpath[b], fr, R[:, 2:5], R[:, 5:9])
            n_samples += len(R)
            if parked.any():
                sc.visibility_samples(bpath[b], fr, ~parked)

    # ---- geoms
    gorder = np.lexsort((G[:, 0], G[:, 1])) if len(G) else np.zeros(0, int)
    Gs = G[gorder] if len(G) else G
    gstarts = np.r_[0, np.flatnonzero(np.diff(Gs[:, 1])) + 1, len(Gs)] if len(G) else [0]
    grows = {int(Gs[a, 1]): Gs[a:b] for a, b in zip(gstarts[:-1], gstarts[1:])} if len(G) else {}
    n_obj = 0
    for g in vg:
        b = int(m.geom_bodyid[g])
        path = f"{bpath[b]}/g{g}"
        R = grows.get(g)
        if R is not None:
            i0 = int(np.searchsorted(R[:, 0], f0, side="right")) - 1
            R = R[max(i0, 0):int(np.searchsorted(R[:, 0], f1, side="left"))]
        gp, gq = (R[0, 2:5], R[0, 5:9]) if R is not None and len(R) and R[0, 0] <= f0 else (None, None)
        p, q = geom_local_original(m, g, gp, gq)
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            f = mesh_gltf(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])))
            if f is None:
                continue
            sc.instance(path, sc.prototype(f, fallback_rgb=_geom_rgb(m, g)), p, q)
        else:
            sc.xform(path, p, q)
            if not _primitive(sc, f"{path}/shape", m, g):
                st.RemovePrim(path)
                continue
        n_obj += 1
        if R is not None and len(R):
            fr = np.maximum(R[:, 0].astype(int) - f0, 0)
            loc = [geom_local_original(m, g, r[2:5], r[5:9]) for r in R]
            sc.vis_attr(path)
            with Sdf.ChangeBlock():
                sc.pose_samples(path, fr, [l[0] for l in loc], [l[1] for l in loc])
                sc.visibility_samples(path, fr, [(r[9] > 0.01 and l[0][2] > -10) for r, l in zip(R, loc)])
        elif gname.startswith("lamp_") and m.geom_rgba[g, 3] < 0.01:
            sc.vis_attr(path, False)

    # ---- cameras of the scene + an overview camera that travels with the vehicle
    cam_paths = {}
    for key, (body, cam) in cams.items():
        c = m.camera(cam).id
        b = 0 if body == "world" else int(m.body(body).id)
        path = f"{bpath[b]}/cam_{key}"
        sc.camera(path, float(m.cam_fovy[c]), m.cam_pos[c], m.cam_quat[c], near=0.03 if key == "wrist" else 0.1)
        cam_paths[key] = path
    vx = z["vehicle_x"][f0:f1]
    has_vehicle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "vehicle") >= 0
    UsdGeom.Scope.Define(st, "/World/cameras")
    if has_vehicle:
        off, tgt = np.array([3.3, 2.7, 2.3]), np.array([1.15, 0.0, 0.55])      # ahead-left of the arm, looking back at it
        cam = sc.camera("/World/cameras/overview", 50.0, off + [float(vx[0]), 0, 0], _look_at_quat(off, tgt))
        tp = Sdf.Path("/World/cameras/overview.xformOp:translate")
        with Sdf.ChangeBlock():
            last = None
            for k, x in enumerate(vx):
                if last is None or abs(x - last) > 1e-4 or k == len(vx) - 1:
                    layer.SetTimeSample(tp, k, Gf.Vec3d(float(off[0] + x), float(off[1]), float(off[2])))
                    last = x
        cam_paths["overview"] = "/World/cameras/overview"

    # ---- decision overlays (T1 / T2 of every target while it is current)
    ov = [(k, d) for k, d in json.loads(str(z["overlays"][0])) if f0 <= k < f1]
    ovl_meta = []
    if ov:
        UsdGeom.Scope.Define(st, "/World/overlay")
        UsdGeom.Scope.Define(st, "/World/overlay_wrist")
    for j, (k0, d) in enumerate(ov):
        k1 = ov[j + 1][0] if j + 1 < len(ov) else f1
        ovl_meta.append(dict(f0=k0 - f0, f1=k1 - f0, text=overlay_text(d)))
        p, w = f"/World/overlay/d{j:03d}", f"/World/overlay_wrist/d{j:03d}"
        sc.xform(p)
        sc.xform(w)
        if d.get("t1") is not None and d.get("t2") is not None:
            sc.sphere(f"{p}/T1", d["t1"], 0.014, (0.1, 0.9, 0.2))
            sc.sphere(f"{p}/T2", d["t2"], 0.014, (0.95, 0.1, 0.1))
            sc.polyline(f"{p}/line", [d["t1"], d["t2"]], 0.004, (1, 1, 1))
        if d.get("window"):
            c = np.array(d["window"])
            sc.polyline(f"{w}/window", np.vstack([c, c[:1]]), 0.003, (1, 1, 1))
        if d.get("bbox"):
            c = np.array(d["bbox"])
            sc.polyline(f"{w}/bbox", np.vstack([c, c[:1]]), 0.002, (0.2, 0.55, 1.0))
        if d.get("shoulder"):
            from pxr import Vt
            X = np.array(d["shoulder"], np.float32).reshape(-1, 3)
            pts = UsdGeom.Points.Define(st, f"{w}/shoulder")
            pts.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(X))
            pts.CreateWidthsAttr([0.007])
            pts.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            pts.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.15, 0.85)])
            pts.CreateExtentAttr([Gf.Vec3f(*(X.min(0) - 0.007).tolist()), Gf.Vec3f(*(X.max(0) + 0.007).tolist())])
        if d.get("candidates"):
            for ci, cpt in enumerate(np.array(d["candidates"]).reshape(-1, 3)[:12]):
                sc.sphere(f"{w}/cand{ci}", cpt, 0.0045, (0.2, 1.0, 1.0))
        for path in (p, w):
            sc.vis_attr(path)
            sc.visibility_samples(path, [0, max(0, k0 - f0), max(0, k1 - f0)], [False, True, False])

    # ---- decoration (Meshcat scenery) and lights
    scn = dict(enabled=False)
    UsdGeom.Scope.Define(st, "/World/lights")
    pano = PROJECT_ROOT / "assets_xgh" / "scenery" / "panorama.jpg"
    yaw_deg = 0.0
    if scenery is not None:
        from pinesim.viz import scenery as S
        p = S.params(scenery)
        bounds = S.field_bounds(m)
        if p.get("enabled", True) and S.available() and bounds is not None:
            sinfo = json.loads((S.DIR / "scenery.json").read_text())
            radius = float(sinfo.get("radius", 40.0))
            mods = S.models()
            lay = S.layout(bounds, p, radius, mods)
            cx, cy = lay["center"]
            yaw_deg = float(p.get("yaw_deg", 0.0))
            UsdGeom.Scope.Define(st, "/World/scenery")
            qyaw = lambda a: [np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)]

            def put(name, file, x, y, zc, yaw, scale=1.0, **kw):
                sc.instance(f"/World/scenery/{name}", sc.prototype(file, **kw), [x, y, zc], qyaw(float(yaw)),
                            None if scale == 1.0 else float(scale))

            put("soil", S.DIR / "soil_disc.gltf", cx, cy, -0.035, 0.0)
            if (S.DIR / "grass_ring.gltf").exists():
                put("grass", S.DIR / "grass_ring.gltf", cx, cy, -0.035, 0.0, drop_alpha_below=0.5, lit=True)
            if sky == "mesh":
                put("dome", S.DIR / "dome.gltf", cx, cy, 0.0, np.radians(yaw_deg), no_shadow=True)
            for i, o in enumerate(lay["rocks"]):
                put(f"rock_{i:02d}", S.DIR / "rock.gltf", o["x"], o["y"], -0.035, o["yaw"], o["scale"])
            for kind in ("trees", "stumps"):
                for i, o in enumerate(lay[kind]):
                    file = (S.MODELS / f"{o['model']}.gltf") if o.get("model") else (S.DIR / "tree.gltf")
                    put(f"{kind}_{i:02d}", file, o["x"], o["y"], -0.035, o["yaw"], o["scale"])
            if lay["hut"] is not None:
                o = lay["hut"]
                put("hut", (S.MODELS / f"{o['model']}.gltf") if o.get("model") else (S.DIR / "hut.gltf"),
                    o["x"], o["y"], -0.035, o["yaw"])
            scn = dict(enabled=True, radius=radius, rocks=len(lay["rocks"]), trees=len(lay["trees"]),
                       stumps=len(lay["stumps"]), hut=lay["hut"] is not None, center=[round(cx, 2), round(cy, 2)], sky=sky)
    dome = UsdLux.DomeLight.Define(st, "/World/lights/sky")
    dome.CreateIntensityAttr(1000.0)
    az0 = _dome_azimuth() if (sky == "light" and scenery is not None and scn.get("enabled") and pano.exists()) else None
    rz = 0.0
    if az0 is not None:
        # lat-long dome light: its pole is the local +Y axis and the middle column of the picture looks along local
        # +Z. rotateX(90) puts the pole on world +Z; the column u then looks at the world azimuth pi/2 - 2 pi u, the
        # Meshcat dome shows it at az0 - 2 pi u -> turn by the difference so that both show the same landscape
        rz = float(np.degrees(az0 - np.pi / 2)) + yaw_deg
        dome.CreateTextureFileAttr(Sdf.AssetPath(sc.texture_file(pano.read_bytes(), "panorama.jpg")))
        dome.CreateTextureFormatAttr("latlong")
    else:
        dome.CreateColorAttr(Gf.Vec3f(0.75, 0.85, 1.0))
    xf = UsdGeom.Xformable(dome.GetPrim())
    xf.AddRotateZOp().Set(rz)
    xf.AddRotateXOp().Set(90.0)
    sun = UsdLux.DistantLight.Define(st, "/World/lights/sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    sx = UsdGeom.Xformable(sun.GetPrim())
    sx.AddRotateZOp().Set(35.0)
    sx.AddRotateXOp().Set(38.0)

    # ---- metadata + side-car file for the video script
    labels = [str(x) for x in z["labels"][f0:f1]]
    runs, a = [], 0
    for k in range(1, len(labels) + 1):
        if k == len(labels) or labels[k] != labels[a]:
            runs.append([a, k, labels[a]])
            a = k
    res = json.loads(str(z["result"][0])) if "result" in z.files else {}
    meta = dict(usd=out_file.name, fps=fps, frames=f1 - f0, duration_s=round((f1 - f0) * dt, 2), frame_range=[f0, f1],
                cameras=cam_paths, labels=runs, overlays=ovl_meta, scenery=scn, info=info or {},
                result={k: res.get(k) for k in ("n_placed", "n_targets", "seed", "field_seed", "policy", "detach_mode",
                                                "sim_time_s", "vehicle_travel_m") if k in res},
                placed_frames=sorted(max(0, int(float(k)) // int(z["every"]) - f0) for k, e in
                                     (x.split(": ", 1) for x in res.get("events", []) if ": " in x)
                                     if e.startswith("fruit in crate") and f0 <= int(float(k)) // int(z["every"]) < f1),
                placed_before=sum(1 for x in res.get("events", []) if ": " in x and x.split(": ", 1)[1].startswith("fruit in crate")
                                  and int(float(x.split(": ", 1)[0])) // int(z["every"]) < f0))
    layer.customLayerData = dict(source="pinesim.viz.usd_export (MuJoCo recording replayed as USD time samples)",
                                 recording=str(Path(rec_path).name), fps=fps, frames=f1 - f0,
                                 cameras=json.dumps(cam_paths))
    layer.comment = "Pineapple Harvest Festival - Mission 3 replay. Visual only (no physics): press Play in Isaac Sim."
    layer.Save()
    meta_file = out_file.with_name(out_file.stem + "_meta.json")
    meta_file.write_text(json.dumps(meta, indent=1))
    return dict(usd=str(out_file), meta=str(meta_file), size_mb=round(out_file.stat().st_size / 1e6, 1), frames=f1 - f0,
                fps=round(fps, 3), bodies=len(bodies), objects=n_obj, prototypes=len(sc.protos), triangles_unique=sc.n_tris,
                pose_samples=n_samples, overlays=len(ov), cameras=list(cam_paths), textures=len(sc.textures), scenery=scn)
