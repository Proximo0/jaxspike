"""Diagnostic: render one frame of a model with the cel ramp alone as its colour (no texture, no lights) to check
that the normal-dot ramp produces bands. blender -b --python _probe_ramp.py -- <model> <out.png>"""
import math
import sys

import bpy
from mathutils import Vector

src, out = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
scene = bpy.context.scene
mat = bpy.data.materials.new("ramp_only")
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
outn = nt.nodes.new("ShaderNodeOutputMaterial")
geo = nt.nodes.new("ShaderNodeNewGeometry")
dot = nt.nodes.new("ShaderNodeVectorMath")
dot.operation = "DOT_PRODUCT"
lv = Vector((-0.6, -0.5, 0.62)).normalized()
dot.inputs[1].default_value = (lv.x, lv.y, lv.z)
half = nt.nodes.new("ShaderNodeMath")
half.operation = "MULTIPLY_ADD"
half.inputs[1].default_value = 0.5
half.inputs[2].default_value = 0.5
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.interpolation = "CONSTANT"
els = ramp.color_ramp.elements
els[0].position = 0.0
els[0].color = (0.2, 0.0, 0.0, 1.0)
els[1].position = 0.42
els[1].color = (0.0, 0.6, 0.0, 1.0)
e = els.new(0.66)
e.color = (0.0, 0.0, 1.0, 1.0)
em = nt.nodes.new("ShaderNodeEmission")
nt.links.new(geo.outputs["Normal"], dot.inputs[0])
nt.links.new(dot.outputs["Value"], half.inputs[0])
nt.links.new(half.outputs["Value"], ramp.inputs[0])
nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
print("ramp elements", [(round(x.position, 2), tuple(round(c, 2) for c in x.color)) for x in els])
print("dot inputs", [i.name for i in dot.inputs], "outputs", [o.name for o in dot.outputs])
print("half inputs", [i.name for i in half.inputs])
for o in bpy.data.objects:
    if o.type == "MESH":
        if o.name.startswith("Icosphere"):
            o.hide_render = True
            continue
        o.data.materials.clear()
        o.data.materials.append(mat)
cam_data = bpy.data.cameras.new("c")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 4.0
cam = bpy.data.objects.new("c", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
elev = math.radians(35)
cam.location = (0.0, -50 * math.cos(elev), 50 * math.sin(elev) + 1.0)
cam.rotation_euler = (math.pi / 2 - elev, 0.0, 0.0)
cam_data.clip_end = 200
scene.render.engine = "BLENDER_EEVEE"
scene.render.film_transparent = True
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.view_settings.view_transform = "Standard"
scene.render.filepath = out
bpy.ops.render.render(write_still=True)
print("done", out)
