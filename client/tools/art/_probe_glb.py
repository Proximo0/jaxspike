"""Diagnostic: import one or more glTF/GLB files in Blender and print mesh, clip, material, normal and transform
facts. Usage: blender -b --python _probe_glb.py -- <file> [<file> ...]"""
import sys

import bpy
from mathutils import Vector

for path in sys.argv[sys.argv.index("--") + 1:]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    print("FILE", path)
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        me = o.data
        det = o.matrix_world.determinant()
        if len(me.vertices) == 0:
            continue
        centre = sum((v.co for v in me.vertices), Vector()) / len(me.vertices)
        poly_score = sum(p.normal.dot(p.center - centre) for p in me.polygons) / max(len(me.polygons), 1)
        vert_score = sum(v.normal.dot(v.co - centre) for v in me.vertices) / len(me.vertices)
        custom = 0.0
        if me.has_custom_normals:
            tot = 0
            for lp in me.loops:
                custom += lp.normal.dot(me.vertices[lp.vertex_index].co - centre)
                tot += 1
            custom /= max(tot, 1)
        mats = [s.material.name if s.material else None for s in o.material_slots]
        print("  MESH %s det=%.2f polys=%d poly_score=%.3f vert_score=%.3f custom=%s custom_score=%.3f mats=%s parent=%s"
              % (o.name, det, len(me.polygons), poly_score, vert_score, me.has_custom_normals, custom, mats,
                 o.parent.name if o.parent else None))
        for mod in o.modifiers:
            print("    MOD", mod.type, mod.name)
    print("  ACTIONS:", sorted(a.name for a in bpy.data.actions)[:12], "...")
    for m in bpy.data.materials:
        if m.node_tree:
            print("  MAT", m.name, [(n.type, getattr(n, "image", None) and n.image.name) for n in m.node_tree.nodes],
                  "blend", getattr(m, "blend_method", None), "backface", m.use_backface_culling)
