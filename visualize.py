import open3d as o3d
import numpy as np

# 读取点云文件
pcd = o3d.io.read_point_cloud("merged_table.ply")

# 删除其中nan的点
pcd = pcd.remove_non_finite_points()

# 获取点云数据（以 numpy 数组形式表示）
# 保存为 .xyz 文件
points = np.asarray(pcd.points)
np.savetxt("merged_table.xyz", points, delimiter=" ", fmt="%.6f", header="x y z", comments="")
# 检查点云总数
print(f"总点数: {len(points)}")

# 可视化
o3d.visualization.draw_geometries([pcd])
