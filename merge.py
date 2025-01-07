import open3d as o3d
import numpy as np
from tqdm import tqdm

def preprocess_point_cloud(pcd, voxel_size):
    """预处理点云：降采样并计算FPFH特征"""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh):
    """使用RANSAC进行全局配准"""
    distance_threshold = 0.05
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result.transformation

def pairwise_registration(source, target):
    """两两配准点云"""
    # 预处理
    voxel_size = 0.01
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    # 全局配准
    init_transform = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh)
    
    # ICP精细配准
    threshold = 0.01
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return reg_p2p.transformation

def load_and_merge_point_clouds():
    """加载并合并点云"""
    pcds = []
    # 读取点云文件
    for i in tqdm(range(8), desc="Loading point clouds"):
        pcd = o3d.io.read_point_cloud(f"data/table_{i+1}.ply")
        pcds.append(pcd)
    
    # 采用连续配准策略
    transformations = [np.identity(4)]
    for i in tqdm(range(1, len(pcds)), desc="Pairwise registration"):
        # 对齐到前一帧
        trans = pairwise_registration(pcds[i], pcds[i-1])
        # 累积变换
        global_trans = transformations[i-1] @ trans
        transformations.append(global_trans)
    
    # 合并点云
    merged_pcd = o3d.geometry.PointCloud()
    for i in tqdm(range(len(pcds)), desc="Merging point clouds"):
        temp = pcds[i]
        temp.transform(transformations[i])
        merged_pcd += temp
    
    # 降采样
    voxel_size = 0.001
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
    
    return merged_pcd

def main():
    # 合并点云
    merged_pcd = load_and_merge_point_clouds()
    
    # 保存结果
    o3d.io.write_point_cloud("merged_table.ply", merged_pcd)
    
    # 可视化
    o3d.visualization.draw_geometries([merged_pcd])

if __name__ == "__main__":
    main()