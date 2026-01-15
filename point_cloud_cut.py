import bpy
import bmesh
import mathutils
import numpy as np
from collections import defaultdict

# =============================================================================
# 1. 网格分割模块 (Mesh Splitter)
# =============================================================================
class MeshSplitter:
    @staticmethod
    def recursive_split(obj, max_verts=100):
        """
        递归分割物体，直到顶点数小于 max_verts
        """
        # 如果不是网格或顶点数达标，停止递归
        if obj.type != 'MESH' or len(obj.data.vertices) <= max_verts:
            return [obj]

        # 1. 决定切割平面 (沿最长边切)
        bbox = [mathutils.Vector(b) for b in obj.bound_box]
        center = sum(bbox, mathutils.Vector()) / 8.0
        dims = obj.dimensions
        
        # 确定法线方向 (切最长的轴)
        if dims.x >= dims.y and dims.x >= dims.z:
            plane_no = (1, 0, 0)
        elif dims.y >= dims.x and dims.y >= dims.z:
            plane_no = (0, 1, 0)
        else:
            plane_no = (0, 0, 1)

        # 2. 进入编辑模式进行 Bisect (切割)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        
        # 使用 Bisect 切割并填补面
        bpy.ops.mesh.bisect(
            plane_co=center,
            plane_no=plane_no,
            use_fill=True,
            clear_inner=False,
            clear_outer=False
        )
        
        # 3. 分离网格 (Separate)
        # 将不相连的部分分离成独立物体
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # 获取分离出的所有物体
        selected_objs = bpy.context.selected_objects
        
        # 如果分离失败（物体是个实心球切不开），强制停止避免死循环
        if len(selected_objs) == 1 and selected_objs[0] == obj:
            return [obj]
        
        # 4. 递归处理子物体并重命名
        final_parts = []
        original_name = obj.name.split("_Part_")[0] # 保持基础名称
        
        for i, part in enumerate(selected_objs):
            # 简单的重命名逻辑
            part.name = f"{original_name}_Part_{i:03d}"
            # 递归
            final_parts.extend(MeshSplitter.recursive_split(part, max_verts))
            
        return final_parts

# =============================================================================
# 2. 形状识别与配准模块 (Signature & Registration)
# =============================================================================
class PointCloudRegistrar:
    
    @staticmethod
    def get_signature(obj):
        """
        获取几何指纹: (顶点数量, 所有顶点到重心的距离之和)
        这对于旋转和平移是不变的。
        """
        if obj.type != 'MESH': return None
        verts = obj.data.vertices
        v_count = len(verts)
        if v_count < 3: return None # 忽略太小的碎片
        
        # 计算局部坐标下的重心
        local_coords = [v.co for v in verts]
        center = sum(local_coords, mathutils.Vector()) / v_count
        
        # 计算距离总和 (保留4位小数作为容差)
        dist_sum = sum((v - center).length for v in local_coords)
        
        return (v_count, round(dist_sum, 4))

    @staticmethod
    def compute_alignment_matrix(source_obj, target_obj):
        """
        计算变换矩阵，使 Source(Local) -> Target(World)。
        这里使用 Kabsch 算法 / SVD。
        """
        # 1. 获取 Source 的局部坐标 (Local Space)
        # 我们希望复用 Source 的 Mesh Data，所以基准是它的 Local 形状
        source_verts = np.array([v.co for v in source_obj.data.vertices])
        
        # 2. 获取 Target 的世界坐标 (World Space)
        # 我们要让 Source 变换后出现在 Target 的位置
        mat_target = target_obj.matrix_world
        target_verts = np.array([mat_target @ v.co for v in target_obj.data.vertices])

        # 3. 计算重心 (Centroids)
        centroid_A = np.mean(source_verts, axis=0) # Source Center (Local)
        centroid_B = np.mean(target_verts, axis=0) # Target Center (World)

        # 4. 归一化点云 (Centering)
        AA = source_verts - centroid_A
        BB = target_verts - centroid_B

        # 5. SVD 计算旋转矩阵
        # H = A^T * B
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        
        # R = V * U^T
        R = np.dot(Vt.T, U.T)

        # 6. 处理反射/镜像情况 (若 Determinant 为负)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # 7. 计算平移向量 t = Centroid_B - R * Centroid_A
        t = centroid_B - np.dot(R, centroid_A)

        # 8. 构建 4x4 变换矩阵
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        
        return mathutils.Matrix(M.tolist())

# =============================================================================
# 3. 主程序 (Main Execution)
# =============================================================================
def optimize_scene():
    # 参数设置
    MAX_VERTS = 100
    
    # 获取当前场景选中的物体，或者所有物体
    # 这里为了安全，我们处理所有选中的Mesh
    objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    
    if not objs:
        print("请先选择要处理的物体！")
        return

    print(f"--- 阶段 1: 递归分割 (阈值: {MAX_VERTS} 顶点) ---")
    
    all_fragments = []
    
    # 必须复制列表，因为我们会动态添加/删除物体
    for obj in objs:
        # 确保应用缩放，否则切割可能会变形
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        fragments = MeshSplitter.recursive_split(obj, max_verts=MAX_VERTS)
        all_fragments.extend(fragments)

    # 刷新场景数据
    bpy.ops.object.select_all(action='DESELECT')
    
    print(f"--- 阶段 2: 几何指纹识别与实例化 (碎片总数: {len(all_fragments)}) ---")
    
    # 按照签名分组
    groups = defaultdict(list)
    
    for obj in all_fragments:
        sig = PointCloudRegistrar.get_signature(obj)
        if sig:
            groups[sig].append(obj)
            
    # 处理每一组
    instanced_count = 0
    
    for sig, group_objs in groups.items():
        if len(group_objs) < 2:
            continue
            
        # 选第一个作为“源物体 (Source)”
        source_obj = group_objs[0]
        
        # 处理其余物体作为“目标 (Target)”
        for target_obj in group_objs[1:]:
            try:
                # 1. 计算对齐矩阵: Source(Local) -> Target(World)
                transform_matrix = PointCloudRegistrar.compute_alignment_matrix(source_obj, target_obj)
                
                # 2. 应用变换
                target_obj.matrix_world = transform_matrix
                
                # 3. 替换数据 (实例化)
                # 重要：将 target 的 Mesh Data 指向 Source 的 Mesh Data
                old_mesh = target_obj.data
                target_obj.data = source_obj.data
                
                # 4. (可选) 清理旧的 Mesh 数据块，防止内存泄漏
                if old_mesh.users == 0:
                    bpy.data.meshes.remove(old_mesh)
                
                instanced_count += 1
                
            except Exception as e:
                print(f"对齐 {target_obj.name} 失败: {e}")

    print(f"--- 优化完成 ---")
    print(f"处理碎片: {len(all_fragments)}")
    print(f"创建实例: {instanced_count}")

if __name__ == "__main__":
    optimize_scene()