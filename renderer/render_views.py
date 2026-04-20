"""
Renders a 3D mesh from multiple synthetic viewpoints using Trimesh (I/O) and
Open3D (rasterization via the native Windows GPU rendering pipeline).

Workflow:
  1. Load an .obj / .glb / .stl / .fbx file.
     .fbx files are loaded directly via Open3D (which bundles assimp).
     All other formats are loaded via Trimesh.
  2. Normalise the mesh: centre on bounding-box midpoint, scale by max extent.
  3. Convert to an Open3D TriangleMesh.
  4. Iterate over every (azimuth, elevation, distance) viewpoint from the grid.
  5. Render each view offscreen and save to  dataset/raw/{object_name}/.

Usage:
    python -m renderer.render_views --config configs/default.yaml --obj models/myobject.obj
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import yaml
from PIL import Image

from .camera_utils import get_sphere_viewpoints, viewpoint_to_camera


def _normalise_vertices(verts: np.ndarray) -> np.ndarray:
    """
    Centre a vertex array on its bounding-box midpoint and scale by max extent.

    Using the bounding-box centre (not mean vertex position) ensures the mesh
    sits symmetrically at the origin regardless of vertex density distribution.
    Dividing by max extent rather than max radius preserves the true aspect
    ratio and guarantees the longest dimension maps to ±0.5.
    """
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    extents = bbox_max - bbox_min

    verts = verts - (bbox_min + bbox_max) / 2.0  # centre at origin

    max_extent = extents.max()
    if max_extent > 0:
        verts /= max_extent

    # Warn if the mesh is suspiciously flat in any axis.
    norm_extents = extents / max_extent if max_extent > 0 else extents
    for axis, label in enumerate("XYZ"):
        if norm_extents[axis] < 0.05:
            print(f"WARNING: mesh is very flat on {label}-axis "
                  f"(normalised extent {norm_extents[axis]:.3f}). "
                  "Some viewpoints may show only a thin slice.")

    return verts


def load_as_open3d(obj_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load a mesh file and return a normalised Open3D TriangleMesh.

    .fbx files are loaded directly via Open3D (which bundles assimp).
    All other formats (.obj with MTL, .glb, .stl, …) are loaded via Trimesh.
    The resulting mesh is centred on its bounding-box midpoint and scaled so
    the longest axis spans [-0.5, 0.5].
    """
    suffix = Path(obj_path).suffix.lower()

    if suffix == ".fbx":
        o3d_mesh = o3d.io.read_triangle_mesh(obj_path)
        if len(o3d_mesh.vertices) == 0:
            raise ValueError(f"Open3D read no vertices from {obj_path}")
        verts = _normalise_vertices(np.asarray(o3d_mesh.vertices).copy())
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    else:
        scene_or_mesh = trimesh.load(obj_path, force="mesh")

        if isinstance(scene_or_mesh, trimesh.Scene):
            meshes = [g for g in scene_or_mesh.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError(f"No triangle geometry found in {obj_path}")
            scene_or_mesh = trimesh.util.concatenate(meshes)

        tm: trimesh.Trimesh = scene_or_mesh
        tm.vertices = _normalise_vertices(tm.vertices.copy())

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(tm.vertices.astype(np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces.astype(np.int32))

    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.6, 0.6, 0.6])
    return o3d_mesh


def render_all_views(config_path: str, obj_path: str) -> None:
    """
    Render a mesh from every viewpoint defined in the config and save PNGs.

    Images are written to:
        <output_path>/raw/<object_stem>/<azimuth>_<elevation>_<distance>.png

    Background is pure white (255, 255, 255) so sky_compositor can cleanly
    detect and replace it using a high-value threshold.

    Args:
        config_path: Path to the YAML config file.
        obj_path:    Path to the mesh file (.obj, .glb, .stl, .fbx).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rcfg = cfg["rendering"]
    img_size: int = rcfg["image_size"]
    obj_name = Path(obj_path).stem

    output_dir = Path(rcfg["output_path"]) / "raw" / obj_name
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_as_open3d(obj_path)

    viewpoints = get_sphere_viewpoints(
        n_angles=rcfg["sphere_angles"],
        distances=rcfg["distances"],
    )

    # Open3D's ViewControl auto-normalises the scene, so changing camera
    # distance has no effect on apparent drone size. Instead we keep the
    # camera fixed and scale the mesh itself — a smaller mesh = drone looks
    # far away; full-size mesh = drone looks close.
    base_distance = min(rcfg["distances"])  # closest distance = full mesh size
    base_verts = np.asarray(mesh.vertices).copy()

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=img_size, height=img_size)
    vis.add_geometry(mesh)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1.0, 1.0, 1.0])  # white background
    render_opt.light_on = True

    saved = 0
    for azimuth, elevation, distance in viewpoints:
        # Scale mesh so it appears at the correct apparent size for this distance.
        scale = base_distance / distance  # e.g. 2.0/12.0 = 0.167 → tiny drone
        mesh.vertices = o3d.utility.Vector3dVector(base_verts * scale)
        mesh.compute_vertex_normals()
        vis.update_geometry(mesh)

        # Camera always at base_distance — only mesh scale changes.
        eye, center, up = viewpoint_to_camera(azimuth, elevation, base_distance)
        front = center - eye
        front /= np.linalg.norm(front)

        vc = vis.get_view_control()
        vc.set_lookat(center.tolist())
        vc.set_front(front.tolist())
        vc.set_up(up.tolist())
        vc.set_zoom(0.45)

        vis.poll_events()
        vis.update_renderer()
        buf = vis.capture_screen_float_buffer(do_render=True)

        img_arr = (np.asarray(buf) * 255).astype(np.uint8)
        fname = output_dir / f"{azimuth:.1f}_{elevation:.1f}_{distance:.2f}.png"
        Image.fromarray(img_arr).save(fname)
        saved += 1

    vis.destroy_window()
    print(f"Rendered {saved} views -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render synthetic views of a 3D mesh")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--obj", required=True,
                        help="Path to mesh file (.obj/.glb/.stl/.fbx)")
    args = parser.parse_args()
    render_all_views(args.config, args.obj)
