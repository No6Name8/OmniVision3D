"""
Renders a 3D mesh from multiple synthetic viewpoints using Trimesh (I/O) and
Open3D (rasterization via the native Windows GPU rendering pipeline).

Workflow:
  1. Load an .obj / .glb / .stl file with Trimesh.
  2. Normalise the mesh to a unit sphere centred at the origin.
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

from .camera_utils import get_viewpoint_grid, viewpoint_to_camera


def load_as_open3d(obj_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load a mesh file via Trimesh and return an Open3D TriangleMesh.

    Trimesh handles .obj (with MTL), .glb, .stl, and many other formats.
    The resulting Open3D mesh is centred at the origin and scaled to fit
    within a sphere of radius 1.
    """
    scene_or_mesh = trimesh.load(obj_path, force="mesh")

    # trimesh.load may return a Scene; merge all geometries into one mesh.
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [g for g in scene_or_mesh.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle geometry found in {obj_path}")
        scene_or_mesh = trimesh.util.concatenate(meshes)

    tm: trimesh.Trimesh = scene_or_mesh

    # Centre and normalise to unit sphere.
    tm.vertices -= tm.vertices.mean(axis=0)
    radius = np.linalg.norm(tm.vertices, axis=1).max()
    if radius > 0:
        tm.vertices /= radius

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tm.vertices.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces.astype(np.int32))
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    return o3d_mesh


def render_all_views(config_path: str, obj_path: str) -> None:
    """
    Render a mesh from every viewpoint defined in the config and save PNGs.

    Images are written to:
        <output_path>/raw/<object_stem>/<azimuth>_<elevation>_<distance>.png

    Args:
        config_path: Path to the YAML config file.
        obj_path:    Path to the mesh file (.obj, .glb, .stl, …).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rcfg = cfg["rendering"]
    img_size: int = rcfg["image_size"]
    obj_name = Path(obj_path).stem

    output_dir = Path(rcfg["output_path"]) / "raw" / obj_name
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_as_open3d(obj_path)

    viewpoints = get_viewpoint_grid(
        azimuth_steps=rcfg["azimuth_steps"],
        elevation_levels=rcfg["elevation_levels"],
        distances=rcfg["distances"],
    )

    # Create one Visualizer and reuse it for all frames.
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=img_size, height=img_size)
    vis.add_geometry(mesh)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.0, 0.0, 0.0])
    render_opt.light_on = True

    saved = 0
    for azimuth, elevation, distance in viewpoints:
        eye, center, up = viewpoint_to_camera(azimuth, elevation, distance)

        # Open3D ViewControl uses a "front" vector = direction from eye to center.
        front = center - eye
        front /= np.linalg.norm(front)

        vc = vis.get_view_control()
        vc.set_lookat(center.tolist())
        vc.set_front(front.tolist())
        vc.set_up(up.tolist())
        # Zoom inversely proportional to distance so the object stays framed.
        vc.set_zoom(0.5 / distance)

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
    parser.add_argument("--obj", required=True, help="Path to mesh file (.obj/.glb/.stl)")
    args = parser.parse_args()
    render_all_views(args.config, args.obj)
