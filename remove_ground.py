import open3d as o3d
import numpy as np
import copy


def fit_plane_ransac(points, distance_threshold=0.01, max_iterations=1000):
    """使用RANSAC拟合平面
    Args:
        points: 地面参考点, shape=(N,3)
        distance_threshold: 点到平面的距离阈值
        max_iterations: 最大迭代次数
    Returns:
        plane_model: [a,b,c,d] 平面方程ax+by+cz+d=0的参数
    """
    best_plane = None
    best_inliers = 0
    
    for _ in range(max_iterations):
        # 随机选择3个点
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]
        
        # 计算平面法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        if np.allclose(normal, 0):
            continue
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, p1)
        
        # 计算所有点到平面的距离
        distances = np.abs(np.dot(points, normal) + d)
        inliers = np.sum(distances < distance_threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = np.append(normal, d)
    
    return best_plane

def remove_ground_by_plane(pcd, ground_points, threshold=0.2):
    """使用拟合的平面移除地面点
    Args:
        pcd: 输入点云
        ground_points: 地面参考点
        threshold: 判定为地面的距离阈值
    """
    # 拟合平面
    plane_model = fit_plane_ransac(ground_points)
    normal = plane_model[:3]
    d = plane_model[3]
    
    # 计算所有点到平面的距离
    points = np.asarray(pcd.points)
    distances = np.abs(np.dot(points, normal) + d)
    
    # 提取非地面点
    mask = distances > threshold
    
    # 创建新点云
    non_ground_pcd = o3d.geometry.PointCloud()
    non_ground_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if pcd.has_normals():
        non_ground_pcd.normals = o3d.utility.Vector3dVector(
            np.asarray(pcd.normals)[mask])
    
    return non_ground_pcd


def sort_by_z(pcd):
    """根据z轴值排序点云"""
    points = np.asarray(pcd.points)
    sort_idx = np.argsort(points[:, 2])
    sorted_points = points[sort_idx]
    sorted_pcd = copy.deepcopy(pcd)
    sorted_pcd.points = o3d.utility.Vector3dVector(sorted_points)
    
    return sorted_points, sorted_pcd


def main(pcd_name):
    # 读取点云文件
    pcd = o3d.io.read_point_cloud(f"lidar_data/{pcd_name}.ply")

    # 删除其中nan的点
    pcd = pcd.remove_non_finite_points()
    o3d.visualization.draw_geometries([pcd])

    # 更新颜色（基于排序后的z值）
    sorted_points, sorted_pcd = sort_by_z(pcd)
    z_coords = sorted_points[:, 2]
    z_normalized = (z_coords - np.min(z_coords)) / (np.max(z_coords) - np.min(z_coords))
    colors = np.zeros((len(sorted_points), 3))
    colors[:, 0] = z_normalized  # R通道
    colors[:, 2] = 1 - z_normalized  # B通道
    sorted_pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化排序后的点云
    o3d.visualization.draw_geometries([sorted_pcd])

    # 从排序好的点云中选取地面点
    while True:
        num_point = int(input("输入地面点数："))
        colors = np.zeros((len(sorted_points), 3))
        colors[:num_point] = [1, 0, 0]
        sorted_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([sorted_pcd])
        if input("是否需要调整地面点数？(y/n)") == "n":
            break
        
    # 使用这些点移除地面
    threshold = 0.4
    while True:
        ground_points = np.asarray(sorted_pcd.points)[:num_point]
        non_ground_pcd = remove_ground_by_plane(sorted_pcd, ground_points, threshold=threshold)
        # 可视化移除地面后的结果
        o3d.visualization.draw_geometries([non_ground_pcd])
        if input("是否需要调整阈值？(y/n)") == "n":
            break
        else:
            threshold = float(input("输入新的阈值："))
        
    # 保存结果
    o3d.io.write_point_cloud(f"lidar_data/nonground_{pcd_name}.ply", non_ground_pcd)
    

if __name__ == "__main__":
    main('sofa_9')