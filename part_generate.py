import open3d as o3d
import numpy as np
import os

import numpy as np
import open3d as o3d

def generate_table_points():
    """生成桌子的点云"""
    points = []
    normals = []
    
    # 桌面参数
    table_length = 1.6  # 长
    table_width = 0.8   # 宽
    table_thickness = 0.04  # 厚度
    height = 0.75       # 高度
    leg_radius = 0.04   # 桌腿半径
    points_per_face = 5000  # 每个面的点数
    
    # 生成桌面点云
    # 上表面
    x = np.random.uniform(-table_length/2, table_length/2, points_per_face)
    y = np.random.uniform(-table_width/2, table_width/2, points_per_face)
    z = np.full_like(x, height)
    points.extend(np.column_stack([x, y, z]))
    normals.extend([[0, 0, 1]] * points_per_face)
    
    # 下表面
    z = np.full_like(x, height - table_thickness)
    points.extend(np.column_stack([x, y, z]))
    normals.extend([[0, 0, -1]] * points_per_face)
    
    # 生成桌腿点云
    leg_positions = [
        [-table_length/2 + leg_radius, -table_width/2 + leg_radius],
        [-table_length/2 + leg_radius, table_width/2 - leg_radius],
        [table_length/2 - leg_radius, -table_width/2 + leg_radius],
        [table_length/2 - leg_radius, table_width/2 - leg_radius]
    ]
    
    points_per_leg = 5000
    for leg_pos in leg_positions:
        # 生成圆柱体点云
        theta = np.random.uniform(0, 2*np.pi, points_per_leg)
        h = np.random.uniform(0, height - table_thickness, points_per_leg)
        x = leg_pos[0] + leg_radius * np.cos(theta)
        y = leg_pos[1] + leg_radius * np.sin(theta)
        z = h
        points.extend(np.column_stack([x, y, z]))
        normals.extend(np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]))
    
    return np.array(points), np.array(normals)

def create_table_surfaces():
    """创建桌子的表面几何形状"""
    surfaces = []
    
    # 桌面参数
    table_length = 1.6
    table_width = 0.8
    height = 0.75
    table_thickness = 0.04
    
    # 桌面(上表面和下表面)
    table_corners = [
        [-table_length/2, -table_width/2],
        [-table_length/2, table_width/2],
        [table_length/2, table_width/2],
        [table_length/2, -table_width/2]
    ]
    
    surfaces.append({
        'corners': table_corners,
        'height': height,
        'normal': [0, 0, 1]
    })
    
    surfaces.append({
        'corners': table_corners,
        'height': height - table_thickness,
        'normal': [0, 0, -1]
    })
    
    return surfaces

def is_point_visible(point, view_position, surfaces):
    """判断点是否可见"""
    # 计算射线方向
    ray_dir = point - view_position
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    
    # 检查射线是否与任何表面相交
    for surface in surfaces:
        # 计算射线与平面的交点
        plane_normal = np.array(surface['normal'])
        plane_point = np.array([surface['corners'][0][0], 
                              surface['corners'][0][1], 
                              surface['height']])
        
        # 射线-平面相交检测
        denom = np.dot(plane_normal, ray_dir)
        if abs(denom) > 1e-6:
            t = np.dot(plane_normal, (plane_point - view_position)) / denom
            if t > 0:  # 交点在射线前方
                intersect = view_position + t * ray_dir
                
                # 检查交点是否在表面范围内
                if is_point_in_polygon(intersect[:2], surface['corners']):
                    # 如果交点到视点的距离小于目标点到视点的距离，则被遮挡
                    if t < np.linalg.norm(point - view_position):
                        return False
    
    return True

def is_point_in_polygon(point, polygon):
    """判断点是否在多边形内"""
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        if (((polygon[i][1] > y) != (polygon[j][1] > y)) and
            (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
             (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
            inside = not inside
        j = i
    
    return inside

def transform_to_local_view(points, normals, view_position):
    """将点云转换到局部视角坐标系"""
    # 计算局部坐标系
    z_axis = -view_position / np.linalg.norm(view_position)  # 指向原点
    temp_up = np.array([0, 0, 1])
    x_axis = np.cross(temp_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # 构建旋转矩阵
    R = np.vstack([x_axis, y_axis, z_axis]).T
    
    # 转换点云到局部坐标系
    local_points = (points - view_position) @ R
    local_normals = normals @ R
    
    return local_points, local_normals

def simulate_view(points, normals, view_position):
    """模拟从特定视角观察到的点云"""
    surfaces = create_table_surfaces()
    visible_points = []
    visible_normals = []
    
    # 先判断可见性
    for point, normal in zip(points, normals):
        if is_point_visible(point, view_position, surfaces):
            visible_points.append(point)
            visible_normals.append(normal)
    
    visible_points = np.array(visible_points)
    visible_normals = np.array(visible_normals)
    
    # 转换到局部坐标系
    local_points, local_normals = transform_to_local_view(
        visible_points, visible_normals, view_position)
    
    return local_points, local_normals

def main():
    # 生成完整桌子点云
    points, normals = generate_table_points()
    
    # 创建完整桌子的点云对象
    complete_pcd = o3d.geometry.PointCloud()
    complete_pcd.points = o3d.utility.Vector3dVector(points)
    complete_pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # 可视化完整桌子点云
    print("显示完整桌子点云...")
    o3d.io.write_point_cloud("sample_table.ply", complete_pcd)
    o3d.visualization.draw_geometries([complete_pcd])
    
    # 定义不同的视角位置
    view_positions = [
        [2, 2, 2],
        [-2, 2, 2],
        [2, -2, 2],
        [-2, -2, 2],
        [2, 2, 0],
        [-2, 2, 0],
        [2, -2, 0],
        [-2, -2, 0]
    ]
    
    # 从每个视角生成部分点云
    for i, view_pos in enumerate(view_positions):
        local_points, local_normals = simulate_view(points, normals, np.array(view_pos))
        
        # 创建点云对象并保存
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(local_points)
        pcd.normals = o3d.utility.Vector3dVector(local_normals)
        os.makedirs("data", exist_ok=True)
        filename = f"data/table_{i+1}.ply"
        o3d.io.write_point_cloud(filename, pcd)
        print(f"保存视角{i+1}的局部坐标系点云，点数 {len(local_points)}")

if __name__ == "__main__":
    main()