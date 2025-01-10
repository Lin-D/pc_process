import open3d as o3d
import numpy as np
from tqdm import tqdm


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0, radius=0.05, min_points=50):
    """移除离群点
    Args:
        pcd: 输入点云
        nb_neighbors: 统计滤波邻居数量
        std_ratio: 统计滤波标准差倍数
        radius: 密度滤波搜索半径
        min_points: 密度滤波最小点数
    """
    # 统计离群值移除
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    
    # 密度滤波
    cl, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
    pcd = pcd.select_by_index(ind)
    
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    """预处理点云: 降采样并计算FPFH特征"""
    # 先移除离群点
    pcd = remove_outliers(pcd, nb_neighbors=15, std_ratio=2.5, radius=0.1, min_points=10)
    # 降采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 3
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 6
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def pairwise_registration(source, target):
    """两两配准点云, 使用双尺度策略"""
    # 双尺度配准参数 - 减少层级避免累积误差
    voxel_sizes = [0.08, 0.04]  # 只用两个尺度
    max_iterations = [50, 100]  # 增加迭代次数
    current_transformation = np.identity(4)
    
    for scale in range(len(voxel_sizes)):
        voxel_size = voxel_sizes[scale]
        iter_num = max_iterations[scale]
        
        # 预处理
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        
        # 全局配准
        distance_threshold = voxel_size * 2.0
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, iter_num))
        
        # ICP精细配准
        current_transformation = result.transformation
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target,  # 使用原始点云
            distance_threshold, 
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iter_num))
        
        current_transformation = reg_p2p.transformation
        print(f"Scale {scale}, Fitness: {reg_p2p.fitness:.4f}")
    
    return current_transformation if reg_p2p.fitness > 0.3 else None


def denoise_merged_cloud(pcd, radius=0.05, knn=30):
    """合并后的点云去噪和优化"""
    # 1. 估计法向量和曲率
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=knn))
    
    # 2. 基于法向量一致性过滤
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    
    # 3. MLS平滑
    alpha = 1.2  # 搜索半径系数
    optimizer = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=radius,
        edge_prune_threshold=0.25,
        reference_node=0)
    
    # 4. 统计滤波去除异常点
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    pcd = pcd.select_by_index(ind)
    
    # 5. RANSAC平面拟合和优化
    smooth_pcd = o3d.geometry.PointCloud()
    smooth_pcd.points = o3d.utility.Vector3dVector(points)
    smooth_pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return smooth_pcd


def load_and_merge_point_clouds():
    """加载并合并点云"""
    pcds = []
    # 读取并预处理点云
    for i in tqdm(range(7), desc="Loading and preprocessing point clouds"):
        pcd = o3d.io.read_point_cloud(f"lidar_data/nonground_table_{i+1}.pcd")
        pcds.append(pcd)
    
    if len(pcds) < 2:
        raise RuntimeError("需要至少两个有效的点云文件")
    
    # 采用连续配准策略
    transformations = [np.identity(4)]
    for i in tqdm(range(1, len(pcds)), desc="Pairwise registration"):
        trans = pairwise_registration(pcds[i], pcds[i-1])
        if trans is None:  # 配准失败
            print(f"跳过第{i}帧点云")
            transformations.append(transformations[-1])  # 使用上一帧变换
            continue
        global_trans = transformations[i-1] @ trans
        transformations.append(global_trans)
    
    # 合并点云
    merged_pcd = o3d.geometry.PointCloud()
    for i in tqdm(range(len(pcds)), desc="Merging point clouds"):
        temp = pcds[i]
        temp.transform(transformations[i])
        merged_pcd += temp
    
    # 最终处理
    merged_pcd = remove_outliers(merged_pcd)  # 移除噪声
    merged_pcd = merged_pcd.voxel_down_sample(0.02)  # 均匀化点云密度
    
    return merged_pcd


def main():
    # 合并点云
    merged_pcd = load_and_merge_point_clouds()
    
    # 保存结果
    o3d.io.write_point_cloud("merged_table.pcd", merged_pcd)
    
    # 可视化
    o3d.visualization.draw_geometries([merged_pcd])

if __name__ == "__main__":
    main()