import open3d as o3d
import numpy as np

def filter_point_cloud(input_path, output_path, max_distance=10.0):
    """基于距离阈值的点云清洗函数
    Params:
        input_path: 输入pcd文件路径
        output_path: 输出pcd文件路径
        max_distance: 最大允许距离阈值（米），参考智能施工设备的有效监测范围设计 [^1]
    """

    pcd = o3d.io.read_point_cloud(input_path)
    
    # 获取坐标数据（Nx3的numpy数组）
    points = np.asarray(pcd.points)
    print(f"原始点云总数: {len(points)}")
    
    # 核心筛选逻辑（向量化计算优化效率）
    z_coords = points[:, 2]
    smallest_z = np.partition(z_coords, 9)[:10]
    z_min = np.mean(smallest_z)
    distances = np.linalg.norm(points, axis=1)  # 欧氏距离计算 [^2]
    valid_mask = (distances > 1e-6) & (distances <= max_distance) & (points[:, 1] < -0.2) & (points[:, 2] < 3) & (points[:, 2] > z_min + 1) # 排除原点及超限点以及挖掘机的一些点
    print(f"清洗后点云总数: {np.sum(valid_mask)}")
    
    # 无有效点时预警输出（类似施工智能监测系统的异常预警机制）
    if not np.any(valid_mask):
        print("警告: 清洗后点云为空，建议调整阈值或检查输入数据")
        return
    
    # 生成清洗后点云（保留原始颜色/法向量等属性）
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[valid_mask])
    if pcd.has_colors():
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[valid_mask])
    if pcd.has_normals():
        filtered_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[valid_mask])
    
    # 保存优化（自动识别文件后缀选择存储格式）
    o3d.io.write_point_cloud(output_path, filtered_pcd, write_ascii=True)


def main():
    folder_path = "test_data"
    file_name = "2"
    input_path = f"{folder_path}/{file_name}.pcd"
    output_path = f"{folder_path}/{file_name}_filtered.pcd"
    filter_point_cloud(input_path, output_path, max_distance=10.0)
    print(f"点云清洗完成，输出路径: {output_path}")
    o3d.visualization.draw_geometries([o3d.io.read_point_cloud(output_path)])
    

if __name__ == "__main__":
    main()
    