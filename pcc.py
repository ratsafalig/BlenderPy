import bpy
import bmesh
import mathutils
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

# =============================================================================
# 0. 配置与数据结构 (Configuration & Data Structures)
# =============================================================================

@dataclass
class ProcessConfig:
    """处理参数配置"""
    max_vertices: int = 100         # 递归分割的阈值
    precision_digits: int = 4       # 指纹匹配的小数精度
    split_padding: float = 0.001    # 切割时的微小容差

class ContextManager:
    """用于安全地切换 Blender 模式的上下文管理器"""
    def __init__(self, mode='OBJECT', active_obj=None):
        self.target_mode = mode
        self.active_obj = active_obj
        self.original_mode = None
        self.original_active = None

    def __enter__(self):
        self.original_mode = bpy.context.object.mode if bpy.context.object else 'OBJECT'
        self.original_active = bpy.context.view_layer.objects.active
        
        if self.active_obj:
            bpy.context.view_layer.objects.active = self.active_obj
            
        if bpy.context.object and bpy.context.object.mode != self.target_mode:
            bpy.ops.object.mode_set(mode=self.target_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始状态，优先恢复物体，再恢复模式
        if self.original_active:
            bpy.context.view_layer.objects.active = self.original_active
        
        try:
            bpy.ops.object.mode_set(mode=self.original_mode)
        except:
            pass # 防止因为物体被删除导致的模式切换失败

# =============================================================================
# 1. 核心数学模块 (Math Solver)
# =============================================================================

class TransformSolver:
    """处理纯数学变换，不依赖 Blender 上下文"""
    
    @staticmethod
    def compute_svd_alignment(source_points: np.ndarray, target_points: np.ndarray) -> mathutils.Matrix:
        """
        使用 SVD (奇异值分解) 计算从 Source 到 Target 的刚体变换矩阵 (4x4)
        
        """
        # 1. 计算质心
        centroid_src = np.mean(source_points, axis=0)
        centroid_tgt = np.mean(target_points, axis=0)

        # 2. 去质心化
        src_centered = source_points - centroid_src
        tgt_centered = target_points - centroid_tgt

        # 3. 计算协方差矩阵 H
        H = np.dot(src_centered.T, tgt_centered)

        # 4. SVD 分解
        U, S, Vt = np.linalg.svd(H)

        # 5. 计算旋转矩阵 R
        R = np.dot(Vt.T, U.T)

        # 6. 修正反射 (确保是右手坐标系旋转，行列式必须为 1)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # 7. 计算平移向量 T
        t = centroid_tgt - np.dot(R, centroid_src)

        # 8. 组装 4x4 矩阵
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t

        return mathutils.Matrix(M.tolist())

# =============================================================================
# 2. 网格处理模块 (Mesh Processor)
# =============================================================================

class MeshPartitioner:
    """负责网格的切割与递归处理"""

    @staticmethod
    def get_split_plane(obj: bpy.types.Object) -> Tuple[mathutils.Vector, Tuple[float, float, float]]:
        """计算最佳切割平面（沿最长轴的中心）"""
        bbox = [mathutils.Vector(b) for b in obj.bound_box]
        center = sum(bbox, mathutils.Vector()) / 8.0
        dims = obj.dimensions
        
        # 简单的轴选择逻辑
        if dims.x >= dims.y and dims.x >= dims.z:
            return center, (1, 0, 0)
        elif dims.y >= dims.x and dims.y >= dims.z:
            return center, (0, 1, 0)
        else:
            return center, (0, 0, 1)

    @classmethod
    def recursive_split(cls, obj: bpy.types.Object, config: ProcessConfig) -> List[bpy.types.Object]:
        """递归分割入口"""
        # 
        
        print(f"递归分割 {cls.name}")

        # 终止条件
        if obj.type != 'MESH' or len(obj.data.vertices) <= config.max_vertices:
            return [obj]

        center, plane_no = cls.get_split_plane(obj)

        # 执行切割
        with ContextManager('EDIT', active_obj=obj):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.bisect(
                plane_co=center,
                plane_no=plane_no,
                use_fill=True,
                clear_inner=False,
                clear_outer=False
            )

        # 执行分离
        # 注意：Separate 会使 obj 变量失效或产生新变量，需重新获取选区
        with ContextManager('OBJECT', active_obj=obj):
            bpy.ops.mesh.separate(type='LOOSE')
        
        selected_objs = bpy.context.selected_objects

        # 检查是否分割失败（例如实心物体未被切开）
        if len(selected_objs) == 1 and selected_objs[0] == obj:
            return [obj]

        # 递归处理子块
        final_fragments = []
        base_name = obj.name.split("_Part_")[0]
        
        for i, part in enumerate(selected_objs):
            part.name = f"{base_name}_Part_{i:03d}"
            # 递归调用
            # final_fragments.extend(cls.recursive_split(part, config))
            
        return final_fragments

# =============================================================================
# 3. 特征分析模块 (Feature Analyzer)
# =============================================================================

class GeometrySignature:
    """生成网格的几何指纹用于匹配"""
    
    @staticmethod
    def compute(obj: bpy.types.Object, precision: int = 4)->int:
        """
        生成指纹: (顶点数, 归一化后的顶点分布特征)
        """
        if obj.type != 'MESH': return None
        verts = obj.data.vertices
        v_count = len(verts)
        if v_count < 3: return None

        # 获取局部坐标点云
        local_coords = np.array([v.co for v in verts])
        
        # 计算局部重心
        center = np.mean(local_coords, axis=0)
        
        # 计算到重心的距离之和 (旋转不变量)
        # 进阶优化：也可以加入惯性张量或表面积作为指纹
        distances = np.linalg.norm(local_coords - center, axis=1)
        dist_sum = np.sum(distances)

        # 保留三位小数
        dist_sum = round(dist_sum, precision)

        # return (v_count, dist_sum)

        ssig = v_count * dist_sum
        ssig = (int)(ssig)

        print(f"GeometrySignature: {v_count}, {dist_sum} {ssig}")

        return ssig

# =============================================================================
# 4. 实例管理模块 (Instance Manager)
# =============================================================================

class InstanceManager:
    """负责将几何体相似的物体替换为实例"""

    @staticmethod
    def replace_with_instance(source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> bool:
        """
        尝试将 target 对齐并替换为 source 的数据
        """
        try:
            # 1. 准备数据
            # Source使用局部坐标 (Local Space)
            src_verts = np.array([v.co for v in source_obj.data.vertices])
            # Target使用世界坐标 (World Space)
            # 
            mat_tgt = target_obj.matrix_world
            tgt_verts = np.array([mat_tgt @ v.co for v in target_obj.data.vertices])

            # 2. 计算变换
            transform_mat = TransformSolver.compute_svd_alignment(src_verts, tgt_verts)

            # 3. 应用变换到 Target
            target_obj.matrix_world = transform_mat

            # 4. 替换 Mesh Data (核心实例化步骤)
            old_mesh = target_obj.data
            target_obj.data = source_obj.data

            # 5. 清理孤立数据
            if old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)

            return True

        except Exception as e:
            print(f"[Error] Failed to instance {target_obj.name}: {e}")
            return False

# =============================================================================
# 5. 主控制器 (Main Controller)
# =============================================================================

class SceneOptimizer:
    def __init__(self, config: ProcessConfig):
        self.config = config

    def run(self):
        # 获取选中的Mesh对象
        initial_objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
        if not initial_objs:
            self.log("未选择任何 Mesh 物体，退出。")
            return

        self.log(f"开始处理: 目标顶点阈值 {self.config.max_vertices}")

        # 1. 预处理：应用缩放 (Scale必须为1，否则SVD计算会有偏差)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        print("开始递归分割")

        # 2. 递归分割
        all_fragments = []
        for obj in initial_objs:
            fragments = MeshPartitioner.recursive_split(obj, self.config)
            all_fragments.extend(fragments)
        
        self.log(f"分割完成，共生成碎片: {len(all_fragments)}")

        # 刷新选区状态，避免干扰
        bpy.ops.object.select_all(action='DESELECT')

        # 3. 指纹分组
        groups = defaultdict(list)
        for frag in all_fragments:
            sig = GeometrySignature.compute(frag, self.config.precision_digits)
            if sig:
                print(f"append {sig} to group")
                groups[sig].append(frag)

        # 4. 执行实例化
        instance_count = 0
        group_count = 0
        
        for sig, objs in groups.items():
            if len(objs) < 2:
                print(f"跳过单个对象组: {sig} {len(objs)}")
                continue
            
            print(f"Group {sig}: {len(objs)}")

            group_count += 1
            source = objs[0] # 该组的"原型"
            targets = objs[1:]

            for target in targets:
                success = InstanceManager.replace_with_instance(source, target)
                if success:
                    instance_count += 1
        
        self.log("--- 优化报告 ---")
        self.log(f"唯一形状组数量: {group_count}")
        self.log(f"成功实例化物体: {instance_count}")
        self.log("完成。")

    def log(self, msg):
        print(f"[SceneOptimizer] {msg}")

# =============================================================================
# 执行入口
# =============================================================================

if __name__ == "__main__":
    # 在这里调整参数
    config = ProcessConfig(
        max_vertices=1000,
        precision_digits=4
    )
    
    optimizer = SceneOptimizer(config)
    optimizer.run()