import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- STRIX HALO APU TRITON BYPASS ---
import triton.runtime.jit
_original_run = triton.runtime.jit.JITFunction.run
def _amd_safe_triton_run(self, *args, **kwargs):
    if kwargs.get('num_warps', 1) > 8: kwargs['num_warps'] = 8
    grid = kwargs.get('grid')
    grid_val = grid(kwargs) if callable(grid) else grid
    if grid_val and grid_val[0] == 0: return 
    return _original_run(self, *args, **kwargs)
triton.runtime.jit.JITFunction.run = _amd_safe_triton_run
# ------------------------------------

import torch
import trimesh
import xatlas
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline

pipeline = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
pipeline.cuda()

# 1. Load the naked geometry
mesh = trimesh.load("assets/example_texturing/the_forgotten_knight.ply")
image = Image.open("assets/example_texturing/image.webp")

# --- THE FIX: UV UNWRAP & ALIGNMENT ---
print("Unwrapping mesh UVs with xatlas...")
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

# 1. Preserve the smooth lighting normals
if hasattr(mesh, 'vertex_normals'):
    mesh.vertex_normals = mesh.vertex_normals[vmapping]

mesh.vertices = mesh.vertices[vmapping]
mesh.faces = indices

# 2. The GLTF V-Flip: Invert the Y-axis so the texture aligns in Blender
uvs[:, 1] = 1.0 - uvs[:, 1]
mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
print("UV unwrap complete!")
# --------------------------------------

# 3. Paint the texture features onto the UV map
output = pipeline.run(mesh, image)
output.export("textured.glb", extension_webp=True)