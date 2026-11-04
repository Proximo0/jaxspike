"""Blender-side renderer for jaxspike 8-direction action sheets (round2 contract 1.4).

Run by `render_kind.py` through `blender -b --python blender_render.py -- <job.json>`; do not run by hand.
The job json (written by render_kind.py from the manifest) holds:
  {"source": "<glb path>", "out": "<dir for per-frame pngs>", "frame": [w, h], "elev_deg": 35,
   "zoom": 1.0, "margin_bottom": 10, "show": [regex...], "hide": [regex...], "tint": [r,g,b] or null,
   "outline": 0.02, "actions": {"idle": {"clip": "Idle", "frames": 4, "loop": true, "hit": null, "span": [0,1]}, ...},
   "dirs": 5, "samples": 16, "supersample": 2}
Each frame is rendered to <out>/<action>_d<dir>_f<frame>.png with a transparent film. Direction 0 faces the
camera (south), each next direction turns the figure 45 degrees clockwise seen from above (SW, W, NW, N ...),
the light stays fixed to the camera so the shading reads the same in every row.
"""
import json
import math
import os
import re
import sys

import bpy
from mathutils import Matrix, Vector

job_path = sys.argv[sys.argv.index("--") + 1]
with open(job_path, "r", encoding="utf-8") as fh:
    JOB = json.load(fh)

FW, FH = JOB["frame"]
SS = int(JOB.get("supersample", 2))
ELEV = math.radians(float(JOB.get("elev_deg", 35.0)))
N_DIRS = int(JOB.get("dirs", 5))


def log(*a):
    print("[render]", *a, flush=True)


# ------------------------------------------------------------------ scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=JOB["source"])

rig = None
meshes = []
for o in list(bpy.data.objects):
    if o.type == "ARMATURE" and rig is None:
        rig = o
    elif o.type == "MESH":
        meshes.append(o)

show = [re.compile(p) for p in JOB.get("show", [])]
hide = [re.compile(p) for p in JOB.get("hide", [])]


def visible(name):
    if any(p.search(name) for p in hide):
        return False
    if show:
        return any(p.search(name) for p in show)
    return True


# Everything hangs under one pivot so the whole figure (rig + loose meshes) turns together.
pivot = bpy.data.objects.new("jx_pivot", None)
scene.collection.objects.link(pivot)
roots = [o for o in bpy.data.objects if o.parent is None and o is not pivot]
for o in roots:
    o.parent = pivot

kept = []
for m in meshes:
    if not visible(m.name):
        m.hide_render = True
        m.hide_viewport = True
    else:
        kept.append(m)
log("meshes kept:", [m.name for m in kept])

# Apply the manifest's model scale (chibi tweaks etc) on the pivot.
msc = float(JOB.get("model_scale", 1.0))
pivot.scale = (msc, msc, msc)


def fix_inward_normals(obj):
    """Some packs (Quaternius Blob family) ship with inward-facing normals: double-sided viewers hide it, but a
    diffuse ramp goes dark and the inverted-hull outline turns inside. Flip when most faces point at the centroid."""
    me = obj.data
    if len(me.polygons) == 0:
        return False
    centre = sum((v.co for v in me.vertices), Vector((0.0, 0.0, 0.0))) / len(me.vertices)
    score = 0.0
    for p in me.polygons:
        score += p.normal.dot(p.center - centre)
    if score >= 0.0:
        return False
    me.flip_normals()
    if me.has_custom_normals:
        me.normals_split_custom_set([(0.0, 0.0, 0.0)] * len(me.loops))
    me.update()
    return True


for m in list(bpy.data.objects):
    if m.type == "MESH" and visible(m.name) and fix_inward_normals(m):
        log("flipped inward normals on", m.name)

# ------------------------------------------------------------------ chibi proportions: scale bones
# "bone_scales": {"head": 1.45, "hand.R": 1.2}: a pose-space scale per bone. Clips that animate that bone's scale
# would override it every frame, so those fcurves are removed from every action first.
BONE_SCALES = JOB.get("bone_scales", {})
def fcurve_groups(act):
    """Every fcurve collection of an action: `act.fcurves` (legacy) or the channelbags of slotted actions (4.4+)."""
    groups = []
    if hasattr(act, "fcurves"):
        groups.append(act.fcurves)
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for cb in getattr(strip, "channelbags", []):
                groups.append(cb.fcurves)
    return groups


if rig is not None and BONE_SCALES:
    removed = 0
    for act in bpy.data.actions:
        for coll in fcurve_groups(act):
            for fc in list(coll):
                for bname in BONE_SCALES:
                    if fc.data_path == 'pose.bones["%s"].scale' % bname:
                        coll.remove(fc)
                        removed += 1
                        break
    log("removed scale fcurves", removed)
    for bname, s in BONE_SCALES.items():
        pb = rig.pose.bones.get(bname)
        if pb is None:
            log("bone_scales: no bone", bname, "have", [b.name for b in rig.pose.bones][:12])
            continue
        pb.scale = (float(s), float(s), float(s))
    log("bone scales", BONE_SCALES)


# ------------------------------------------------------------------ face decal
def bone_world(bone_name):
    pb = rig.pose.bones.get(bone_name)
    if pb is None:
        return None
    mw = rig.matrix_world
    return mw @ pb.head, mw @ pb.tail


def make_face(spec):
    """A subdivided plane with the face image, parented to the head bone and shrink-wrapped onto the head mesh so it
    hugs the surface and turns with the head in every direction. `spec`: {"image", "bone", "target" (regex over
    mesh names), "size" (fraction of the model height), "up" (shift along the bone, fraction), "forward"}."""
    import bmesh
    img_path = spec["image"]
    if not os.path.isfile(img_path):
        log("face: no image", img_path)
        return None
    bone = spec.get("bone", "head")
    bw = bone_world(bone) if rig is not None else None
    tgt_re = re.compile(spec.get("target", ".*"))
    target = None
    for o in kept:
        if tgt_re.search(o.name):
            target = o
            break
    if target is None or bw is None:
        log("face: no target/bone", spec.get("target"), bone)
        return None
    head, tail = bw
    centre = head + (tail - head) * float(spec.get("up", 0.5))
    size = float(spec.get("size", 0.22)) * height
    # the plane starts in front of the target's bounding box; shrinkwrap pulls it onto the surface
    dg2 = bpy.context.evaluated_depsgraph_get()
    ev = target.evaluated_get(dg2)
    ymin = min((ev.matrix_world @ v.co).y for v in ev.data.vertices)
    me = bpy.data.meshes.new("jx_face_mesh")
    bm = bmesh.new()
    n = 12
    verts = [[bm.verts.new((-size / 2 + size * i / n, 0.0, -size / 2 + size * j / n)) for i in range(n + 1)] for j in range(n + 1)]
    for j in range(n):
        for i in range(n):
            bm.faces.new((verts[j][i], verts[j][i + 1], verts[j + 1][i + 1], verts[j + 1][i]))
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        for lp in f.loops:
            co = lp.vert.co
            lp[uv_layer].uv = ((co.x + size / 2) / size, (co.z + size / 2) / size)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new("jx_face", me)
    scene.collection.objects.link(obj)
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone
    obj.matrix_world = Matrix.Translation(Vector((centre.x, ymin - 0.05 * height, centre.z + float(spec.get("lift", 0.0)) * height)))
    # face -Y (the figure's front) : the plane was built in the XZ plane with its normal along -Y already? bmesh
    # face winding above gives +Y or -Y; flip to -Y if needed
    me.calc_normals_split() if hasattr(me, "calc_normals_split") else None
    if me.polygons and me.polygons[0].normal.y > 0:
        me.flip_normals()
    sw = obj.modifiers.new("jx_wrap", "SHRINKWRAP")
    sw.target = target
    # nearest-surface wrap: direction-free (a bone-parented child's local axes are the bone's, so a projection
    # along "its" Y would shoot along the bone and miss); each vertex snaps to the closest head surface point
    sw.wrap_method = "NEAREST_SURFACEPOINT"
    sw.offset = float(spec.get("offset", 0.012)) * height
    sw.wrap_mode = "ABOVE_SURFACE"
    mat = bpy.data.materials.new("jx_face_mat")
    nt = mat.node_tree
    for nn in list(nt.nodes):
        nt.nodes.remove(nn)
    o2 = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(img_path)
    tex.image.colorspace_settings.name = "sRGB"
    em = nt.nodes.new("ShaderNodeEmission")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], o2.inputs["Surface"])
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
        try:
            setattr(mat, attr, val)
        except Exception:  # noqa: BLE001
            pass
    mat.use_backface_culling = True
    me.materials.append(mat)
    log("face decal", spec.get("style", ""), "on", target.name, "bone", bone, "size", round(size, 3))
    return obj



# ------------------------------------------------------------------ toon material
TINT = JOB.get("tint")
RAMP = JOB.get("ramp", [[0.0, 0.62], [0.42, 0.84], [0.72, 1.0]])
OUTLINE = float(JOB.get("outline", 0.02))


def toonify(mat):
    """Rewrite a material to a 3-step cel ramp over its base colour (texture or flat), emission output."""
    nt = mat.node_tree
    if nt is None:
        mat.use_nodes = True
        nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    # find a base colour source
    tex = None
    base_rgb = (0.8, 0.8, 0.8, 1.0)
    for n in nodes:
        if n.type == "TEX_IMAGE" and n.image is not None:
            tex = n
        if n.type == "BSDF_PRINCIPLED":
            base_rgb = tuple(n.inputs["Base Color"].default_value)
    tex_name = tex.name if tex is not None else None
    for n in list(nodes):
        if n.name != tex_name:
            nodes.remove(n)
    tex = nodes.get(tex_name) if tex_name else None  # removal invalidates earlier node references
    out = nodes.new("ShaderNodeOutputMaterial")
    # Light term: dot(world normal, fixed light vector) mapped to 0..1. Scene lights are not used at all, so the
    # cel steps are deterministic (no shadow noise) and read the same in every direction; the figure rotates under
    # a light that stays fixed to the screen, like a hand-painted sprite.
    geo = nodes.new("ShaderNodeNewGeometry")
    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    ld = JOB.get("light_dir", [-0.6, -0.5, 0.62])  # towards the light: screen upper-left, in front of the figure
    lv = Vector((float(ld[0]), float(ld[1]), float(ld[2]))).normalized()
    dot.inputs[1].default_value = (lv.x, lv.y, lv.z)
    half = nodes.new("ShaderNodeMath")
    half.operation = "MULTIPLY_ADD"
    half.inputs[1].default_value = 0.5
    half.inputs[2].default_value = 0.5
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    els = ramp.color_ramp.elements
    while len(els) > 1:
        els.remove(els[-1])
    els[0].position = RAMP[0][0]
    els[0].color = (RAMP[0][1],) * 3 + (1.0,)
    for pos, v in RAMP[1:]:
        e = els.new(pos)
        e.color = (v, v, v, 1.0)
    mul = nodes.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    links.new(geo.outputs["Normal"], dot.inputs[0])
    links.new(dot.outputs["Value"], half.inputs[0])
    links.new(half.outputs["Value"], ramp.inputs[0])
    if tex is not None:
        tex.interpolation = "Closest"
        src = tex.outputs["Color"]
    else:
        rgb = nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = base_rgb
        src = rgb.outputs[0]
    if TINT:
        tn = nodes.new("ShaderNodeMix")
        tn.data_type = "RGBA"
        tn.blend_type = "MULTIPLY"
        tn.inputs["Factor"].default_value = float(JOB.get("tint_amount", 1.0))
        tn.inputs[7].default_value = (TINT[0], TINT[1], TINT[2], 1.0)
        links.new(src, tn.inputs[6])
        src = tn.outputs[2]
    # saturation / brightness push for the bright anime look
    hsv = nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = float(JOB.get("saturation", 1.25))
    hsv.inputs["Value"].default_value = float(JOB.get("brightness", 1.08))
    links.new(src, hsv.inputs["Color"])
    src = hsv.outputs["Color"]
    lift = JOB.get("lift")  # additive [r, g, b]: lightens a dark palette entry towards a pastel (multiply cannot)
    if lift:
        ln = nodes.new("ShaderNodeMix")
        ln.data_type = "RGBA"
        ln.blend_type = "ADD"
        ln.inputs["Factor"].default_value = 1.0
        ln.inputs[7].default_value = (float(lift[0]), float(lift[1]), float(lift[2]), 1.0)
        links.new(src, ln.inputs[6])
        src = ln.outputs[2]
    links.new(src, mul.inputs[6])
    links.new(ramp.outputs["Color"], mul.inputs[7])
    col = mul.outputs[2]
    # specular highlight: a hard white-ish spot where the light term is near 1
    spec_amt = float(JOB.get("specular", 0.35))
    if spec_amt > 0:
        sp = nodes.new("ShaderNodeMath")
        sp.operation = "GREATER_THAN"
        sp.inputs[1].default_value = float(JOB.get("specular_at", 0.93))
        links.new(half.outputs["Value"], sp.inputs[0])
        spm = nodes.new("ShaderNodeMix")
        spm.data_type = "RGBA"
        spm.blend_type = "MIX"
        spm.inputs[7].default_value = (1.0, 1.0, 1.0, 1.0)
        spf = nodes.new("ShaderNodeMath")
        spf.operation = "MULTIPLY"
        spf.inputs[1].default_value = spec_amt
        links.new(sp.outputs["Value"], spf.inputs[0])
        links.new(spf.outputs["Value"], spm.inputs[0])
        links.new(col, spm.inputs[6])
        col = spm.outputs[2]
    # rim light: a pale band where the surface turns away from the camera (silhouette edges), screen-fixed
    rim_amt = float(JOB.get("rim", 0.3))
    if rim_amt > 0:
        facing = nodes.new("ShaderNodeVectorMath")
        facing.operation = "DOT_PRODUCT"
        links.new(geo.outputs["Normal"], facing.inputs[0])
        links.new(geo.outputs["Incoming"], facing.inputs[1])
        fa = nodes.new("ShaderNodeMath")
        fa.operation = "ABSOLUTE"
        links.new(facing.outputs["Value"], fa.inputs[0])
        rt = nodes.new("ShaderNodeMath")
        rt.operation = "LESS_THAN"
        rt.inputs[1].default_value = float(JOB.get("rim_at", 0.3))
        links.new(fa.outputs["Value"], rt.inputs[0])
        rf = nodes.new("ShaderNodeMath")
        rf.operation = "MULTIPLY"
        rf.inputs[1].default_value = rim_amt
        links.new(rt.outputs["Value"], rf.inputs[0])
        rc = JOB.get("rim_color", [0.85, 0.9, 1.0])
        rmx = nodes.new("ShaderNodeMix")
        rmx.data_type = "RGBA"
        rmx.blend_type = "ADD"
        rmx.inputs[7].default_value = (float(rc[0]), float(rc[1]), float(rc[2]), 1.0)
        links.new(rf.outputs["Value"], rmx.inputs[0])
        links.new(col, rmx.inputs[6])
        col = rmx.outputs[2]
    links.new(col, emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mat.use_backface_culling = False


outline_mat = bpy.data.materials.new("jx_outline")
onodes = outline_mat.node_tree.nodes
for n in list(onodes):
    onodes.remove(n)
o_out = onodes.new("ShaderNodeOutputMaterial")
o_em = onodes.new("ShaderNodeEmission")
oc = JOB.get("outline_color", [0.08, 0.05, 0.1])
o_em.inputs["Color"].default_value = (oc[0], oc[1], oc[2], 1.0)
outline_mat.node_tree.links.new(o_em.outputs["Emission"], o_out.inputs["Surface"])
outline_mat.use_backface_culling = True

# ------------------------------------------------------------------ bounds (rest pose) for framing
dg = bpy.context.evaluated_depsgraph_get()
zmin, zmax = 1e9, -1e9
rmax = 0.0
for m in kept:
    ev = m.evaluated_get(dg)
    for v in ev.data.vertices:
        p = ev.matrix_world @ v.co
        zmin = min(zmin, p.z)
        zmax = max(zmax, p.z)
        rmax = max(rmax, math.hypot(p.x, p.y))
height = max(zmax - zmin, 0.1)
log("bounds z", round(zmin, 3), round(zmax, 3), "height", round(height, 3), "radius", round(rmax, 3))

# outline thickness: "outline_rel" is a fraction of the model height (thicker, darker lines read as anime ink);
# "outline" (world units) is the older absolute form
if "outline_rel" in JOB:
    OUTLINE = float(JOB["outline_rel"]) * height
done_mats = set()
for m in kept:
    for slot in m.material_slots:
        if slot.material is not None and slot.material.name not in done_mats:
            toonify(slot.material)
            done_mats.add(slot.material.name)
    if OUTLINE > 0:
        m.data.materials.append(outline_mat)
        mod = m.modifiers.new("jx_outline", "SOLIDIFY")
        mod.thickness = -OUTLINE
        mod.offset = 1.0
        mod.use_flip_normals = True
        mod.use_rim = False
        mod.material_offset = len(m.data.materials) - 1
        mod.material_offset_rim = len(m.data.materials) - 1

FACE = JOB.get("face")
face_obj = None
if FACE and rig is not None:
    try:
        face_obj = make_face(FACE)
    except Exception as exc:  # noqa: BLE001
        log("face decal failed:", repr(exc))
# "floor": put the lowest rest-pose vertex on the ground (hovering models and off-origin pivots), plus an
# optional lift in model heights for things that should hover a little
if JOB.get("floor", False):
    pivot.location.z = -zmin * msc + float(JOB.get("hover", 0.0)) * height

# ------------------------------------------------------------------ camera + light
cam_data = bpy.data.cameras.new("jx_cam")
cam_data.type = "ORTHO"
cam = bpy.data.objects.new("jx_cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
dist = 50.0
cam.location = (0.0, -dist * math.cos(ELEV), dist * math.sin(ELEV))
cam.rotation_euler = (math.pi / 2 - ELEV, 0.0, 0.0)
cam_data.clip_end = 200.0
# frame fit: the figure's projected height ~ height*cos(elev) + base depth*sin(elev); fit it into FH less margins.
# The depth uses a capped radius (a held sword or a long tail would otherwise shrink the whole figure).
zoom = float(JOB.get("zoom", 1.0))
mb = float(JOB.get("margin_bottom", 10))
mt = float(JOB.get("margin_top", 6))
r_fit = min(rmax, height * float(JOB.get("radius_cap", 0.3)))
proj_h = height * math.cos(ELEV) + 2.0 * r_fit * math.sin(ELEV)
fit_px = FH - mb - mt
upp = proj_h / fit_px / zoom  # world units per output pixel
cam_data.ortho_scale = upp * max(FW, FH)
# shift_y moves the view up (positive) so the ground origin lands `mb` px above the bottom edge; the near half of
# the base ring (r_fit*sin(elev)) hangs below the origin, so part of that depth is given back with base_lift.
base_px = r_fit * math.sin(ELEV) / upp * float(JOB.get("base_lift", 0.6))
cam_data.shift_y = (FH / 2.0 - mb - base_px) / max(FW, FH)
ANCHOR = [FW / 2.0, FH - mb]
log("ortho_scale", round(cam_data.ortho_scale, 4), "shift_y", round(cam_data.shift_y, 4), "anchor", ANCHOR)

sun_data = bpy.data.lights.new("jx_sun", "SUN")
sun_data.energy = float(JOB.get("sun", 3.0))
sun = bpy.data.objects.new("jx_sun", sun_data)
scene.collection.objects.link(sun)
# key light travelling from the screen's upper left towards lower right and away from the camera, at a real side
# angle, so the right and lower parts of a figure fall into the shade steps (a sun near the camera axis lights every
# visible face the same and the cel ramp shows nothing)
sd = JOB.get("sun_dir", [0.65, 0.45, -0.6])
sun.rotation_euler = Vector((float(sd[0]), float(sd[1]), float(sd[2]))).to_track_quat("-Z", "Y").to_euler()
scene.world = bpy.data.worlds.new("jx_world")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (0.35, 0.35, 0.4, 1.0)
    bg.inputs[1].default_value = float(JOB.get("ambient", 0.5))

# ------------------------------------------------------------------ render settings
scene.render.engine = "BLENDER_EEVEE"
scene.render.film_transparent = True
scene.render.resolution_x = FW * SS
scene.render.resolution_y = FH * SS
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.compression = 15
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.display_settings.display_device = "sRGB"
ee = scene.eevee
ee.taa_render_samples = int(JOB.get("samples", 8))
for attr, val in (("use_gtao", False), ("use_bloom", False), ("use_shadows", True)):
    if hasattr(ee, attr):
        setattr(ee, attr, val)
scene.render.use_motion_blur = False
scene.render.fps = 24

# ------------------------------------------------------------------ animation sampling
actions_by_name = {a.name: a for a in bpy.data.actions}


def set_action(name):
    act = actions_by_name.get(name)
    if act is None:
        raise SystemExit("no clip %r in %s; have %s" % (name, JOB["source"], sorted(actions_by_name)))
    if rig.animation_data is None:
        rig.animation_data_create()
    rig.animation_data.action = act
    # Blender 4.4+ slotted actions: pick the first slot that fits the rig
    try:
        if hasattr(act, "slots") and len(act.slots) and hasattr(rig.animation_data, "action_slot"):
            rig.animation_data.action_slot = act.slots[0]
    except Exception as exc:  # noqa: BLE001
        log("slot assign failed", exc)
    return act


def sample_times(n, f0, f1, loop, hit, span):
    """n evenly spaced clip frames over [f0, f1]; loops leave out the end (it equals the start);
    `hit` (clip fraction of the contact moment) is warped onto sample index round(0.6*(n-1))."""
    a, b = span
    lo = f0 + (f1 - f0) * a
    hi = f0 + (f1 - f0) * b
    if n <= 1:
        return [lo]
    out = []
    for i in range(n):
        t = i / (n if loop else n - 1)
        if hit is not None and 0.0 < hit < 1.0:
            hi_i = round(0.6 * (n - 1))
            if i <= hi_i:
                t = hit * (i / max(hi_i, 1))
            else:
                t = hit + (1.0 - hit) * ((i - hi_i) / max(n - 1 - hi_i, 1))
        out.append(lo + (hi - lo) * t)
    return out


out_dir = JOB["out"]
total = 0
for aname, spec in JOB["actions"].items():
    act = set_action(spec["clip"])
    f0, f1 = act.frame_range
    frames = sample_times(int(spec["frames"]), f0, f1, bool(spec.get("loop", False)), spec.get("hit"),
                          spec.get("span", [0.0, 1.0]))
    squash = float(JOB.get("squash", 0.0)) if aname in ("idle", "walk") else 0.0
    for d in range(N_DIRS):
        # direction 0 faces the camera (-Y is toward the camera, glTF +Z forward becomes Blender -Y)
        pivot.rotation_euler = (0.0, 0.0, -math.radians(45.0 * d) + math.radians(float(spec.get("yaw", JOB.get("yaw", 0.0)))))
        for i, ft in enumerate(frames):
            scene.render.filepath = "%s/%s_d%d_f%02d.png" % (out_dir, aname, d, i)
            if os.path.isfile(scene.render.filepath):
                continue  # resumed run: keep what an earlier (crashed) run already wrote
            # squash-and-stretch for blobs: a bounce cycle over the clip, volume roughly kept
            if squash:
                ph = 2.0 * math.pi * i / max(len(frames), 1) * (2 if aname == "walk" else 1)
                sq = squash * math.sin(ph)
                pivot.scale = (msc * (1.0 + sq), msc * (1.0 + sq), msc * (1.0 - 2.0 * sq))
            else:
                pivot.scale = (msc, msc, msc)
            scene.frame_set(int(math.floor(ft)), subframe=float(ft - math.floor(ft)))
            bpy.ops.render.render(write_still=True)
            total += 1
    log("action", aname, "clip", spec["clip"], "frames", [round(x, 2) for x in frames])

meta = {"anchor": ANCHOR, "frame": [FW, FH], "supersample": SS, "rendered": total, "height": height}
with open(out_dir + "/_meta.json", "w", encoding="utf-8") as fh:
    json.dump(meta, fh)
log("done", total, "frames")
