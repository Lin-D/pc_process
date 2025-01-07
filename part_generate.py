import open3d as o3d
import numpy as np
import math

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
    points_per_face = 20000  # 每个面的点数
    
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
    
    points_per_leg = 15000
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

def simulate_view(points, normals, view_position):
    """模拟从特定视角观察到的点云"""
    surfaces = create_table_surfaces()
    visible_points = []
    visible_normals = []
    
    for point, normal in zip(points, normals):
        if is_point_visible(point, view_position, surfaces):
            visible_points.append(point)
            visible_normals.append(normal)
    
    return np.array(visible_points), np.array(visible_normals)

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
        [2, 2, 2],     # 右上方
        [-2, 2, 2],    # 左上方
        [2, -2, 2],    # 右下方
        [-2, -2, 2],   # 左下方
        [0, 0, 3],     # 正上方
        [2, 0, 1],     # 右侧中部
        [-2, 0, 1],    # 左侧中部
        [0, 2, 1],     # 前方中部
        [0, -2, 1],    # 后方中部
        [0, 0, 0.5],   # 低角度
        [0, 0, -1],    # 从下方看
        [1, 1, -1],
        [2, 1, -0.5],
        [0, -2, -1],
        [-1, -1, -0.5]
    ]
    
    # 从每个视角生成部分点云
    for i, view_pos in enumerate(view_positions):
        visible_points, visible_normals = simulate_view(points, normals, np.array(view_pos))
        
        # 创建点云对象并保存
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(visible_points)
        pcd.normals = o3d.utility.Vector3dVector(visible_normals)
        
        filename = f"data/table_{i+1}.ply"
        o3d.io.write_point_cloud(filename, pcd)
        print(f"保存视角{i+1}的点云，视点位置 {view_pos}，点数 {len(visible_points)}")

if __name__ == "__main__":
    main()