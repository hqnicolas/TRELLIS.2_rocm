from typing import *
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual
from flex_gemm.ops.grid_sample import grid_sample_3d
try:
    import nvdiffrast.torch as dr
except ImportError:
    dr = None
import cumesh

try:
    from trellis2.utils.pipeline_logger import get_logger, log_mesh, log_uv, log_tensor, elapsed, section
    _HAS_LOGGER = True
except Exception:
    _HAS_LOGGER = False
    def get_logger(): return None
    def log_mesh(*a, **kw): pass
    def log_uv(*a, **kw): pass
    def log_tensor(*a, **kw): pass
    def elapsed(): return ""
    def section(t): pass


def _L():
    return get_logger() if _HAS_LOGGER else None

def _info(msg):
    L = _L()
    if L: L.info(msg)

def _debug(msg):
    L = _L()
    if L: L.debug(msg)

def _error(msg):
    L = _L()
    if L: L.error(msg)

def _log_cumesh(mesh, tag):
    try:
        v, f = mesh.read()
        log_mesh(v, f, tag)
    except Exception as e:
        _error(f"  [cumesh:{tag}] read failed: {e}")


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Convert an extracted mesh to a GLB file.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.
    
    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sprase tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
    """
    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    
    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3
    
    section("to_glb: START")
    _info(f"  {elapsed()}  to_glb  decimation_target={decimation_target}  texture_size={texture_size}  remesh={remesh}")
    log_mesh(vertices, faces, "to_glb:input")
    _info(f"  {elapsed()}  aabb={aabb.tolist() if hasattr(aabb, 'tolist') else aabb}")
    log_tensor(attr_volume, "attr_volume")
    log_tensor(coords, "coords")

    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Move data to GPU
    vertices = vertices.cuda()
    faces = faces.cuda()

    # Initialize CUDA mesh handler
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)

    # --- Initial Mesh Cleaning ---
    # Fills holes as much as we can before processing
    _info(f"  {elapsed()}  cumesh: fill_holes (initial)")
    mesh.fill_holes(max_hole_perimeter=3e-2)
    _log_cumesh(mesh, "after-initial-fill_holes")
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    vertices, faces = mesh.read()
    if use_tqdm:
        pbar.update(1)
        
    # Build BVH for the current mesh to guide remeshing
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose:
        print(f"Building BVH for current mesh...", end='', flush=True)
    _info(f"  {elapsed()}  cumesh: building BVH  vertices={vertices.shape}  faces={faces.shape}")
    bvh = cumesh.cuBVH(vertices, faces)
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    if verbose:
        print("Cleaning mesh...")

    # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
    if not remesh:
        section("to_glb: simplify+clean (standard)")

        # Step 1: Aggressive simplification (3x target)
        _info(f"  {elapsed()}  cumesh: simplify(3x={decimation_target * 3})")
        mesh.simplify(decimation_target * 3, verbose=verbose)
        _log_cumesh(mesh, "after-simplify-3x")
        if verbose:
            print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        _info(f"  {elapsed()}  cumesh: remove_duplicate_faces")
        mesh.remove_duplicate_faces()
        _info(f"  {elapsed()}  cumesh: repair_non_manifold_edges")
        mesh.repair_non_manifold_edges()
        _info(f"  {elapsed()}  cumesh: remove_small_connected_components(1e-5)")
        mesh.remove_small_connected_components(1e-5)
        _info(f"  {elapsed()}  cumesh: fill_holes (after initial cleanup)")
        mesh.fill_holes(max_hole_perimeter=3e-2)
        _log_cumesh(mesh, "after-initial-cleanup")
        if verbose:
            print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 3: Final simplification to target count
        _info(f"  {elapsed()}  cumesh: simplify(final={decimation_target})")
        mesh.simplify(decimation_target, verbose=verbose)
        _log_cumesh(mesh, "after-final-simplify")
        if verbose:
            print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 4: Final Cleanup loop
        _info(f"  {elapsed()}  cumesh: final cleanup pass")
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        _log_cumesh(mesh, "after-final-cleanup")
        if verbose:
            print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Step 5: Unify face orientations
        _info(f"  {elapsed()}  cumesh: unify_face_orientations")
        mesh.unify_face_orientations()
        _log_cumesh(mesh, "after-unify-orientations")

    # --- Branch 2: Remeshing Pipeline ---
    else:
        section("to_glb: remesh pipeline")
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        _info(f"  {elapsed()}  remesh: center={center.tolist()}  scale={scale:.4g}  resolution={resolution}")

        # Perform Dual Contouring remeshing (rebuilds topology)
        _info(f"  {elapsed()}  cumesh: remesh_narrow_band_dc")
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = verbose,
            bvh = bvh,
        ))
        _log_cumesh(mesh, "after-remesh-dc")
        if verbose:
            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

        # Simplify and clean the remeshed result (similar logic to above)
        _info(f"  {elapsed()}  cumesh: simplify(remesh={decimation_target})")
        mesh.simplify(decimation_target, verbose=verbose)
        _log_cumesh(mesh, "after-remesh-simplify")
        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    
    # --- UV Parameterization ---
    section("to_glb: UV unwrap")
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")

    _info(f"  {elapsed()}  cumesh: uv_unwrap  cone_half_angle={np.degrees(mesh_cluster_threshold_cone_half_angle_rad):.1f}deg")
    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()

    log_mesh(out_vertices, out_faces, "uv-unwrap:out_vertices/faces")
    log_uv(out_uvs, "uv-unwrap:out_uvs")
    _info(f"  {elapsed()}  uv_unwrap: out_vmaps shape={list(out_vmaps.shape)}  "
          f"min={out_vmaps.min().item()}  max={out_vmaps.max().item()}")

    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]
    log_tensor(out_normals, "uv-unwrap:out_normals")

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Baking (Attribute Sampling) ---
    section("to_glb: texture baking")
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)

    # Setup differentiable rasterizer context
    ctx = dr.RasterizeCudaContext()
    # Prepare UV coordinates for rasterization (rendering in UV space)
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    _info(f"  {elapsed()}  uvs_rast: shape={list(uvs_rast.shape)}  "
          f"xy=[{uvs_rast[0,:,0].min():.4g},{uvs_rast[0,:,0].max():.4g}] x "
          f"[{uvs_rast[0,:,1].min():.4g},{uvs_rast[0,:,1].max():.4g}]")

    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)

    # Rasterize in chunks to save memory
    n_chunks = (out_faces.shape[0] + 99999) // 100000
    _info(f"  {elapsed()}  rasterize-chunks: {out_faces.shape[0]} faces  n_chunks={n_chunks}  texture={texture_size}x{texture_size}")
    total_covered = 0
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i+100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        covered = int(mask_chunk.sum().item())
        total_covered += covered
        _debug(f"  {elapsed()}  rast chunk {i//100000}: covered={covered}")
        rast_chunk[..., 3:4] += i # Store face ID in alpha channel
        rast = torch.where(mask_chunk, rast_chunk, rast)

    # Mask of valid pixels in texture
    mask = rast[0, ..., 3] > 0
    n_valid = int(mask.sum().item())
    total_pixels = texture_size * texture_size
    _info(f"  {elapsed()}  rast final: valid_pixels={n_valid}/{total_pixels} ({100.*n_valid/total_pixels:.1f}%)  "
          f"rast_alpha range=[{rast[0,...,3].min():.4g}, {rast[0,...,3].max():.4g}]")
    if n_valid == 0:
        _error(f"  {elapsed()}  ⚠ ZERO valid texels after rasterization — texture will be all black!")

    # Interpolate 3D positions in UV space (finding 3D coord for every texel)
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    _info(f"  {elapsed()}  interpolated valid_pos: shape={list(valid_pos.shape)}  "
          f"x=[{valid_pos[:,0].min():.4g},{valid_pos[:,0].max():.4g}]  "
          f"y=[{valid_pos[:,1].min():.4g},{valid_pos[:,1].max():.4g}]  "
          f"z=[{valid_pos[:,2].min():.4g},{valid_pos[:,2].max():.4g}]")

    # Map these positions back to the *original* high-res mesh to get accurate attributes
    # This corrects geometric errors introduced by simplification/remeshing
    _info(f"  {elapsed()}  BVH unsigned_distance: querying {valid_pos.shape[0]} points")
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    _info(f"  {elapsed()}  BVH result: face_id range=[{face_id.min().item()},{face_id.max().item()}]  "
          f"faces_available={faces.shape[0]}")
    if face_id.max().item() >= faces.shape[0]:
        _error(f"  {elapsed()}  ⚠ BVH face_id OUT OF BOUNDS — face_id.max={face_id.max().item()}  faces={faces.shape[0]}")
    orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    _info(f"  {elapsed()}  BVH-corrected valid_pos: "
          f"x=[{valid_pos[:,0].min():.4g},{valid_pos[:,0].max():.4g}]  "
          f"y=[{valid_pos[:,1].min():.4g},{valid_pos[:,1].max():.4g}]  "
          f"z=[{valid_pos[:,2].min():.4g},{valid_pos[:,2].max():.4g}]")

    # Trilinear sampling from the attribute volume (Color, Material props)
    grid_coords = ((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3)
    _info(f"  {elapsed()}  grid_sample_3d: grid_coords range "
          f"x=[{grid_coords[0,:,0].min():.4g},{grid_coords[0,:,0].max():.4g}]  "
          f"y=[{grid_coords[0,:,1].min():.4g},{grid_coords[0,:,1].max():.4g}]  "
          f"z=[{grid_coords[0,:,2].min():.4g},{grid_coords[0,:,2].max():.4g}]  "
          f"aabb={aabb.tolist()}  voxel_size={voxel_size.tolist()}  grid_size={grid_size.tolist()}")
    out_of_bounds_frac = ((grid_coords < -1).any(-1) | (grid_coords > 1).any(-1)).float().mean().item()
    if out_of_bounds_frac > 0:
        _error(f"  {elapsed()}  ⚠ {out_of_bounds_frac*100:.1f}% of grid sample coords are out of [-1,1] — expect clamped/wrong texture!")

    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    if mask.any(): attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=grid_coords,
        mode='trilinear',
    )
    _info(f"  {elapsed()}  attrs: shape={list(attrs.shape)}  "
          f"min={attrs.min():.4g}  max={attrs.max():.4g}  mean={attrs.mean():.4g}  "
          f"NaN={torch.isnan(attrs).any().item()}  inf={torch.isinf(attrs).any().item()}")
    for ch_name, ch_slice in attr_layout.items():
        ch = attrs[..., ch_slice]
        _info(f"  {elapsed()}  attrs[{ch_name}]: min={ch.min():.4g}  max={ch.max():.4g}  mean={ch.mean():.4g}")

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Post-Processing & Material Construction ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)
    
    mask = mask.cpu().numpy()
    
    # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha_mode = 'OPAQUE'
    
    # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    
    # Create PBR material
    # Standard PBR packs Metallic and Roughness into Blue and Green channels
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True if not remesh else False,
    )
    
    # --- Coordinate System Conversion & Final Object ---
    section("to_glb: finalize trimesh")
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()

    _info(f"  {elapsed()}  pre-axis-swap: vertices x=[{vertices_np[:,0].min():.4g},{vertices_np[:,0].max():.4g}]  "
          f"y=[{vertices_np[:,1].min():.4g},{vertices_np[:,1].max():.4g}]  "
          f"z=[{vertices_np[:,2].min():.4g},{vertices_np[:,2].max():.4g}]")
    _info(f"  {elapsed()}  uvs (pre-flip): u=[{uvs_np[:,0].min():.4g},{uvs_np[:,0].max():.4g}]  "
          f"v=[{uvs_np[:,1].min():.4g},{uvs_np[:,1].max():.4g}]")

    # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1].copy()
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1].copy()
    uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate

    _info(f"  {elapsed()}  post-axis-swap: vertices x=[{vertices_np[:,0].min():.4g},{vertices_np[:,0].max():.4g}]  "
          f"y=[{vertices_np[:,1].min():.4g},{vertices_np[:,1].max():.4g}]  "
          f"z=[{vertices_np[:,2].min():.4g},{vertices_np[:,2].max():.4g}]")
    _info(f"  {elapsed()}  uvs (post-flip): u=[{uvs_np[:,0].min():.4g},{uvs_np[:,0].max():.4g}]  "
          f"v=[{uvs_np[:,1].min():.4g},{uvs_np[:,1].max():.4g}]")

    n_nan_v = int(np.isnan(vertices_np).any(axis=1).sum())
    n_nan_uv = int(np.isnan(uvs_np).any(axis=1).sum())
    if n_nan_v or n_nan_uv:
        _error(f"  {elapsed()}  ⚠ NaN in final output: vertices={n_nan_v}  uvs={n_nan_uv}")
    _info(f"  {elapsed()}  final trimesh: {vertices_np.shape[0]} vertices  {faces_np.shape[0]} faces  {uvs_np.shape[0]} uvs")

    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
    )

    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")

    section("to_glb: DONE")
    return textured_mesh