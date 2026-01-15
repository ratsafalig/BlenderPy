import bpy
from mathutils import Vector
import time

# --- 全局约束配置 ---
# 只有顶点数大于此阈值的松散组件才会被分离成新的 Object 进行实例化。
MIN_VERTICES_FOR_SEPARATION = 100 
# --------------------

def geometry_hash(obj):
    """
    创建一个基于标准化几何特征的Hash签名 (Invariant to Baked Scale)。
    """
    if obj.type != 'MESH' or not obj.data.vertices:
        return None
    
    mesh = obj.data
    v_count = len(mesh.vertices)
    p_count = len(mesh.polygons)
    
    if v_count == 0:
        return None
    
    # 确保 Mesh 是更新过的
    mesh.update() 
    
    # 修正：通过 Object 获取 dimensions
    try:
        max_dim = obj.dimensions.length 
    except AttributeError:
        # 如果获取失败，跳过
        return None

    if max_dim < 1e-4:
        return None
        
    # 计算网格中心点 (局部空间)
    bbox_center = sum((v.co for v in mesh.vertices), Vector()) / v_count
    
    # 提取并标准化关键顶点 (只取前10个)
    normalized_coords_tuple = []
    for i in range(min(10, v_count)):
        v = mesh.vertices[i]
        normalized_co = ((v.co - bbox_center) / max_dim).to_tuple()
        normalized_coords_tuple.append(normalized_co)
        
    # 排序以忽略可能的顶点顺序打乱
    normalized_coords_tuple.sort() 

    normalized_hash = hash(tuple(normalized_coords_tuple)) 
    
    return f"{v_count}_{p_count}"


def deduplicate_meshes(scope_objects):
    """
    执行 Mesh 数据去重并修复世界变换 (Visual Transform)。
    返回 (已实例化的物体数量, 唯一的 Mesh 数据块列表)
    """
    unique_meshes = {} # Store {hash: mesh_data}
    instanced_count = 0

    # 确保在 Object 模式下，这样 matrix_world 才是准确的
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    for obj in scope_objects:
        
        original_world_matrix = obj.matrix_world.copy()
        g_hash = geometry_hash(obj)
        
        if g_hash is None:
             continue
        
        if g_hash in unique_meshes:
            target_mesh = unique_meshes[g_hash]
            
            if obj.data != target_mesh:
                old_mesh = obj.data
                
                # 核心：关联到共享 Mesh 数据块
                obj.data = target_mesh
                instanced_count += 1
                
                # 核心：修复视觉变换
                obj.matrix_world = original_world_matrix 
                
                # 清理旧 Mesh
                if old_mesh.users == 0:
                    bpy.data.meshes.remove(old_mesh, do_unlink=True)
                
        else:
            unique_meshes[g_hash] = obj.data
            
    return instanced_count, unique_meshes.values()


def separate_and_filter_parts(obj, min_verts):
    """
    对单个 Object 执行按松散部分分离，并过滤掉顶点数过小的部分，将其重新 Join 回原 Object。
    【改进】为新 Object 设置父级关系和关联名称。
    """
    # 确保是 Mesh Object 且有顶点
    if obj.type != 'MESH' or not obj.data.vertices:
        return 0
    
    # 1. 记录分离前的 Object 集合
    objects_before = set(bpy.context.scene.objects)
    
    # 2. 执行按松散部分分离的准备工作
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    try:
        # 进入编辑模式并分离
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.separate(type='LOOSE')
    except Exception as e:
        # 无法进入编辑模式或分离失败，跳过
        print(f"分离 {obj.name} 失败: {e}")
        return 0
    finally:
        # 无论成功与否，都要返回 Object 模式
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # 3. 识别新创建的 Object
    objects_after = set(bpy.context.scene.objects)
    new_parts = list(objects_after - objects_before)
    
    parts_separated_count = len(new_parts)
    small_parts_to_join = []
    
    # 4. 【核心改进】设置父级和名称，并检查约束
    index = 0
    for part in new_parts:
        if part.type == 'MESH':
            
            # A. 设置父级：将新 Object 放在原 Object 之下
            part.parent = obj
            
            # B. 重命名：关联原 Object 的名称
            part.name = f"{obj.name}_Part.{index:03d}"
            index += 1
            
            # 过滤逻辑：收集顶点数过小的组件
            if len(part.data.vertices) <= min_verts:
                small_parts_to_join.append(part)

    # 5. 将顶点数过小的组件重新 Join 回原 Object
    if small_parts_to_join:
        bpy.ops.object.select_all(action='DESELECT')
        
        # 将原 Object 设置为 Join 的目标 (active)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # 选中所有小组件
        for part in small_parts_to_join:
            # Join 操作会自动移除这些 Object，并销毁其父级关系
            part.select_set(True)

        # 执行 Join
        bpy.ops.object.join()
        
        print(f"  > {obj.name}: Join 回了 {len(small_parts_to_join)} 个小组件。")

    return parts_separated_count

# --- 优化阶段 ---

def phase_one_object_deduplication():
    """
    阶段 1：扫描现有 Object，执行快速实例化。
    最高优先级的优化。
    """
    print("=========================================")
    print("--- 阶段 1: 现有 Object 快速去重 (最高优先级) ---")
    
    # 仅处理现有 Mesh Object
    initial_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    
    start_time = time.time()
    instanced_count, _ = deduplicate_meshes(initial_objects)
    
    print(f"阶段 1 完成。用时 {time.time() - start_time:.2f} 秒。")
    print(f"成功实例链接 {instanced_count} 个现有 Object。")
    return instanced_count


def phase_two_granular_optimization():
    """
    阶段 2：执行带约束的组件分离，并对新 Object 再次去重。
    """
    print("\n=========================================")
    print(f"--- 阶段 2: 内部组件细化分离与去重 (Min Verts: {MIN_VERTICES_FOR_SEPARATION}) ---")
    
    # 复制列表，因为分离操作会创建新 Object
    target_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    total_parts_separated = 0
    
    start_time = time.time()
    
    for obj in target_objects:
        # 如果 Object 已经被实例链接 (users > 1)，通常不需要分离其内部组件，跳过
        if obj.data.users > 1: 
            continue
            
        parts_count = separate_and_filter_parts(obj, MIN_VERTICES_FOR_SEPARATION)
        if parts_count > 0:
            total_parts_separated += parts_count
            
    # 分离完成后，对场景中所有 Mesh Object (包括新的) 再次运行去重
    new_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    instanced_count, _ = deduplicate_meshes(new_objects)
    
    print(f"阶段 2 完成。用时 {time.time() - start_time:.2f} 秒。")
    print(f"总共分离并处理了 {total_parts_separated} 个组件。")
    print(f"成功实例链接 {instanced_count} 个组件 Object。")
    print("=========================================")


def run_full_optimization():
    """执行完整的两阶段优化流程"""
    print("\n\n--- 🤖 启动两阶段 Mesh 实例化优化 ---")
    
    # 确保当前模式正确
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
        
    # Phase 1: 现有 Object 去重
    phase_one_object_deduplication()
    
    # Phase 2: 组件细化与去重
    phase_two_granular_optimization()
    
    # 最终清理
    bpy.ops.object.select_all(action='DESELECT')
    
    print("\n✅ 整体优化流程执行完毕。请导出 FBX 文件并检查 Unity 中的 Draw Call 和内存占用。")
    
# 执行完整的优化
run_full_optimization()