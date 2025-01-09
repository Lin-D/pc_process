import open3d as o3d
import numpy as np

original_pcd = o3d.io.read_point_cloud("sample_table.ply")
merged_pcd = o3d.io.read_point_cloud("merged_table.ply")

aabb = merged_pcd.get_axis_aligned_bounding_box()
obb = merged_pcd.get_oriented_bounding_box()
aabb.color = (1, 0, 0)
obb.color = (0, 1, 0)

o3d.visualization.draw_geometries([merged_pcd, aabb, obb])
