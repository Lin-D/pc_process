import open3d as o3d
import numpy as np
from tqdm import tqdm
import copy


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
    # voxel_sizes=[0.5, 0.2, 0.1, 0.05]  # office parameters
    voxel_sizes=[2, 1, 0.5, 0.2, 0.1]
    max_iterations=[100, 100, 100, 100, 100]
    current_transformation = np.identity(4)
    
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    
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
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iter_num))
        
        current_transformation = reg_p2p.transformation
        # 打印最小尺度下的配准匹配度
        if scale == len(voxel_sizes) - 1:
            best_fitness = reg_p2p.fitness
            print(f"Final Fitness: {best_fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}")
    
    return current_transformation if reg_p2p.fitness > 0.3 else None


def denoise_merged_cloud(pcd, radius=0.1, knn=50):
    """合并后的点云去噪和优化
    Args:
        pcd: 输入点云
        radius: 搜索半径, 增大以获取更大区域
        knn: k近邻数量, 增大以考虑更多点
    """
    # 1. 估计法向量 - 使用更大的搜索范围
    search_param = o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius*1.5, max_nn=knn)
    pcd.estimate_normals(search_param)
    
    # 2. 统计滤波 - 放宽过滤条件
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=100,  # 增大邻域点数
        std_ratio=2.0     # 放宽标准差倍数
    )
    pcd = pcd.select_by_index(ind)
    
    # 3. 平滑处理
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    smoothed_points = np.zeros_like(points)
    
    # 对每个点进行加权平均平滑
    for i in range(len(points)):
        # 查找大范围邻域点
        [k, idx, _] = pcd_tree.search_radius_vector_3d(points[i], radius)
        if k < 10:  # 确保有足够的邻域点
            smoothed_points[i] = points[i]
            continue
            
        # 基于距离的加权平均
        neighbors = points[idx]
        distances = np.linalg.norm(neighbors - points[i], axis=1)
        weights = np.exp(-distances / (radius * 0.5))  # 高斯权重
        smoothed_points[i] = np.average(neighbors, weights=weights, axis=0)
    
    # 创建平滑后的点云
    smooth_pcd = o3d.geometry.PointCloud()
    smooth_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
    smooth_pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return smooth_pcd


def load_and_merge_point_clouds(pcd_num):
    """加载并合并点云"""
    pcds = []
    # 读取并预处理点云
    for i in tqdm(range(1, pcd_num+1), desc="Loading point clouds"):
        pcd = o3d.io.read_point_cloud(f"test_data/{i}_filtered.pcd")
        # pcd = remove_outliers(pcd, nb_neighbors=15, std_ratio=2.5, radius=0.1, min_points=10)
        pcds.append(pcd)
    
    # 分组处理
    # 将所有的点云分为多个组，每组选取一个基准帧
    # 组内其余帧向基准帧进行配准，然后再进行组间基准帧的配准
    group_size = 20
    # 如果所提供的帧数太少，直接进行顺序帧间配准
    if pcd_num < group_size:
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd += pcds[0]
        if pcd_num == 1:
            print("Only one frame, no need to group")
        else:
            print("Too few frames to group, change registration strategy to frame-wise registration")
            transforms = [np.identity(4)]
            for i in tqdm(range(1, len(pcds)), desc='Frame-wise Registration'):
                trans = pairwise_registration(pcds[i], pcds[i-1])
                if trans is None:
                    transforms.append(transforms[-1])
                else:
                    global_trans = transforms[-1] @ trans
                    transforms.append(global_trans)
                temp = copy.deepcopy(pcds[i])
                temp.transform(transforms[i])
                merged_pcd += temp
            
    else:
        groups = []
        ref_pcds = []  # 每组的参考帧
        group_transforms = []  # 每组内部的变换矩阵
        
        # 1. 分组并处理每个组
        for i in range(0, len(pcds), group_size):
            group = pcds[i:i+group_size]
            if len(group) < 2:
                continue
                
            # 选择组中间帧作为参考
            ref_idx = len(group) // 2
            ref_pcd = group[ref_idx]
            ref_pcds.append(ref_pcd)
            
            # 组内配准，调用pairwise_registration函数
            transforms = [np.identity(4)] * len(group)
            for j in tqdm(range(len(group)), desc=f'Intra-group Registration of Group No.{i//group_size+1} of {len(pcds)//group_size+1}'):
                if j != ref_idx:
                    trans = pairwise_registration(group[j], ref_pcd)
                    if trans is not None:
                        transforms[j] = trans
                    else:
                        print(f"Failed to register frame {j} to reference frame {ref_idx} in group {i//group_size+1}")
            
            groups.append(group)
            group_transforms.append(transforms)
        
        # 2. 基准帧间配准
        ref_transforms = [np.identity(4)]
        for i in tqdm(range(1, len(ref_pcds)), desc='Ref-frame Registration'):
            trans = pairwise_registration(ref_pcds[i], ref_pcds[i-1])
            if trans is None:
                ref_transforms.append(ref_transforms[-1])
            else:
                global_trans = ref_transforms[-1] @ trans
                ref_transforms.append(global_trans)
        
        # 3. 合并所有点云
        merged_pcd = o3d.geometry.PointCloud()
        for i, group in enumerate(groups):
            ref_trans = ref_transforms[i]
            group_trans = group_transforms[i]
            
            # 对每一帧，先应用组内变换，再应用参考帧变换，完成全局变换
            for j, pcd in enumerate(group):
                temp = copy.deepcopy(pcd)
                temp.transform(group_trans[j])
                temp.transform(ref_trans)
                merged_pcd += temp
    
    # 最终处理
    # merged_pcd = remove_outliers(merged_pcd)
    # merged_pcd = merged_pcd.voxel_down_sample(0.02)
    
    return merged_pcd


def main():
    # 合并点云
    merged_pcd = load_and_merge_point_clouds(pcd_num=5)
    
    # 保存结果
    o3d.io.write_point_cloud("merged/merged_truck4.pcd", merged_pcd)
    
    # 可视化
    o3d.visualization.draw_geometries([merged_pcd], window_name="Merged Point Cloud")

if __name__ == "__main__":
    main()
    