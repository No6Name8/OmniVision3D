"""Renderer package — synthetic view generation from 3D meshes using Trimesh + Open3D."""

from .camera_utils import get_viewpoint_grid, viewpoint_to_camera
from .render_views import render_all_views

__all__ = ["get_viewpoint_grid", "viewpoint_to_camera", "render_all_views"]
