from __future__ import annotations

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from isaaclab.sim.utils import clone, get_current_stage

from .assembly_geometry import hull_mesh

MATERIAL_ROOT = "/World/Materials"


def physics_material(name: str, friction: float, combine: str = "average") -> UsdShade.Material:
    stage = get_current_stage()
    path = f"{MATERIAL_ROOT}/{name}"
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        return UsdShade.Material(prim)
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction))
    api.CreateRestitutionAttr(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr(combine)
    px.CreateRestitutionCombineModeAttr("min")
    return mat


def bind_physics_material(prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _visual_material(stage, path: str, colour) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in colour]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


@clone
def spawn_convex_body(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.XformCommonAPI(root)
    if translation is not None:
        xf.SetTranslate(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        x, y, z, w = (float(v) for v in orientation)
        root.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    for i, pts in enumerate(cfg.pieces):
        verts, faces = hull_mesh(pts)
        name = cfg.piece_names[i] if cfg.piece_names else f"piece_{i:03d}"
        mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*(float(c) for c in v)) for v in verts]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(k) for k in faces.ravel()]))
        mesh.CreateSubdivisionSchemeAttr("none")
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim).CreateHullVertexLimitAttr(64)
        col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        col.CreateContactOffsetAttr(float(cfg.contact_offset))
        col.CreateRestOffsetAttr(0.0)
    if cfg.mass > 0.0:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(cfg.mass))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
        rb.CreateSolverPositionIterationCountAttr(int(cfg.position_iterations))
        rb.CreateSolverVelocityIterationCountAttr(1)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
    bind_physics_material(root.GetPrim(), physics_material(cfg.material_name, cfg.friction))
    vis = _visual_material(stage, f"{prim_path}/Looks/colour", cfg.colour)
    UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(vis)
    return root.GetPrim()


def filter_contacts(a_root: str, b_root: str, rules: dict) -> int:
    stage = get_current_stage()
    a, b = stage.GetPrimAtPath(a_root), stage.GetPrimAtPath(b_root)
    if not a.IsValid() or not b.IsValid():
        raise RuntimeError(f"filter_contacts: {a_root} or {b_root} does not exist")
    b_pieces = [p for p in b.GetChildren() if p.HasAPI(UsdPhysics.CollisionAPI)]
    n = 0
    for piece in a.GetChildren():
        if not piece.HasAPI(UsdPhysics.CollisionAPI):
            continue
        for prefix, others in rules.items():
            if piece.GetName().startswith(prefix):
                rel = UsdPhysics.FilteredPairsAPI.Apply(piece).CreateFilteredPairsRel()
                for other in b_pieces:
                    if other.GetName().startswith(tuple(others)):
                        rel.AddTarget(other.GetPath())
                        n += 1
    return n
