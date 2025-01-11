import open3d as o3d
import numpy as np


def optimize_planar_region(pcd, distance_threshold=0.02, min_ratio=0.05):
    points = np.asarray(pcd.points)
    optimized_points = points.copy()

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    remaining_indices = list(range(len(points)))
    while len(remaining_indices) > len(points) * min_ratio:
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points[remaining_indices])

        plane_model, inliers = temp_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) < len(remaining_indices) * min_ratio:
            break
    
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)  # 单位化法向量

        plane_indices = np.array(remaining_indices)[inliers]
        plane_points = points[plane_indices]

        # 计算点到平面的投影
        distances = np.dot(plane_points, normal) + d
        projections = plane_points - np.outer(distances, normal)

        # 优化后的点
        optimized_points[plane_indices] = projections

        # 去掉已处理的点
        remaining_indices = list(set(remaining_indices) - set(plane_indices))

    optimized_pcd = o3d.geometry.PointCloud()
    optimized_pcd.points = o3d.utility.Vector3dVector(optimized_points)

    return optimized_pcd


for i in range(20):
    pcd = o3d.io.read_point_cloud(f"lidar_data/table_{i+1}.pcd")
    optimized_pcd = optimize_planar_region(pcd, distance_threshold=0.02, min_ratio=0.1)
    o3d.io.write_point_cloud(f"lidar_data/optimized_table_{i+1}.pcd", optimized_pcd)