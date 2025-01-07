import open3d as o3d
import numpy as np

def pairwise_registration(source, target):
    """两两配准点云"""
    threshold = 0.02
    # 直接使用ICP进行配准
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

def load_and_merge_point_clouds():
    """加载并合并点云"""
    pcds = []
    # 读取点云文件
    for i in range(15):
        pcd = o3d.io.read_point_cloud(f"data/table_{i+1}.ply")
        pcds.append(pcd)
    
    # 以第一个点云为基准，依次配准
    transformations = [np.identity(4)]
    for i in range(1, len(pcds)):
        trans = pairwise_registration(pcds[i], pcds[0])
        transformations.append(trans)
    
    # 合并点云
    merged_pcd = o3d.geometry.PointCloud()
    for i in range(len(pcds)):
        temp = pcds[i]
        temp.transform(transformations[i])
        merged_pcd += temp
    
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