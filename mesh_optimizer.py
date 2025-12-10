import bpy
import bmesh
import mathutils
from collections import defaultdict
import numpy as np

class MeshUtils:

    @staticmethod
    def get_verts_world_space(obj):

        mat_world = obj.matrix_world

        mesh = obj.data

        verts_world = []

        for v in mesh.vertices:

            world_coord = mat_world @ v.co

            verts_world.append(world_coord)

        return verts_world

    @staticmethod
    def get_verts_local_space(obj):

        return [v.co.copy() for v in obj.data.vertices]

    @staticmethod
    def get_center_of_mass(verts):
        if not verts:
            return mathutils.Vector((0, 0, 0))
        return sum(verts, mathutils.Vector()) / len(verts)

class MeshSignature:
    @staticmethod
    def get_signature(obj):

        if obj.type != "MESH" or obj.data is None:
            return None

        verts = MeshUtils.get_verts_local_space(obj)

        vert_count = len(verts)

        if vert_count == 0:
            return (0, 0.0)

        center = MeshUtils.get_center_of_mass(verts)

        dist_sum = sum((v - center).length for v in verts)

        result = (vert_count, round(dist_sum, 4))

        print(f"Signature for {obj.name}: {result}")

        return result
    
class InstanceAligner:
    @staticmethod
    def get_alignment_basis_pca(obj):
        verts_world_space = MeshUtils.get_verts_world_space(obj)

        center = MeshUtils.get_center_of_mass(verts_world_space)
    
        verts_count = len(verts_world_space)

        if verts_count < 3:
            return None
        
        verts_np = np.array([v - center.to_tuple() for v in verts_world_space])

        center_np = np.mean(verts_np, axis=0)
        center_blender = mathutils.Vector(center_np)

        P_centered = verts_np - center_np

        U, S, Vh = np.linalg.svd(P_centered, full_matrices=False)

        vec_x_np = Vh[0, :]
        vec_y_np = Vh[1, :]
        vec_z_np = Vh[2, :]

        vec_z_recalc_np = np.cross(vec_x_np, vec_y_np)

        if np.dot(vec_z_recalc_np, vec_z_np) < 0:
            vec_x_np = -vec_x_np
            vec_z_recalc_np = np.cross(vec_x_np, vec_y_np)

        vec_x = mathutils.Vector(vec_x_np)
        vec_y = mathutils.Vector(vec_y_np)
        vec_z = mathutils.Vector(vec_z_recalc_np)

        rot_matrix = mathutils.Matrix([vec_x, vec_y, vec_z]).transposed()

        return center_blender, rot_matrix

    @staticmethod
    def align_object(source_obj, target_obj):

        source_res = InstanceAligner.get_alignment_basis_pca(source_obj)
        target_res = InstanceAligner.get_alignment_basis_pca(target_obj)

        if not source_res or not target_res:
            print(f"Skipping alignment for {target_obj.name}: Not enough vertices.")
            return

        s_center, s_rot = source_res
        t_center, t_rot = target_res

        center_diff = t_center - s_center
        rotation_diff = t_rot @ s_rot.inverted()

        matrix_diff = rotation_diff.to_4x4()

        matrix_diff[0][3] = center_diff.x
        matrix_diff[1][3] = center_diff.y
        matrix_diff[2][3] = center_diff.z

        target_obj.matrix_world = target_obj.matrix_world @ matrix_diff

        target_obj.data = source_obj.data


class MeshSplitter:

    @staticmethod
    def recursive_split(obj, max_verts=100, original_name=None):

        if original_name is None:
            original_name = obj.name

        if len(obj.data.vertices) <= max_verts:
            return [obj]

        bbox = [mathutils.Vector(b) for b in obj.bound_box]
        center = sum(bbox, mathutils.Vector()) / 8
        dimensions = obj.dimensions

        if dimensions.x >= dimensions.y and dimensions.x >= dimensions.z:
            plane_no = (1, 0, 0)
        elif dimensions.y >= dimensions.x and dimensions.y >= dimensions.z:
            plane_no = (0, 1, 0)
        else:
            plane_no = (0, 0, 1)

        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")

        bpy.ops.mesh.bisect(
            plane_co=center,
            plane_no=plane_no,
            use_fill=True,
            clear_inner=False,
            clear_outer=False,
        )

        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")

        selected_objects = bpy.context.selected_objects

        if len(selected_objects) == 1 and selected_objects[0] == obj:
            print(f"Cannot split solid mesh {obj.name} further with simple separation.")
            return [obj]

        result_objects = []

        for part in selected_objects:
            result_objects.extend(
                MeshSplitter.recursive_split(part, max_verts, original_name)
            )

        return result_objects

    @staticmethod
    def process_object(obj, threshold=100):
        if obj.type != "MESH":
            return

        if len(obj.data.vertices) <= threshold:
            return

        print(f"Processing Large Object: {obj.name}")

        original_name = obj.name
        original_matrix = obj.matrix_world.copy()

        parts = MeshSplitter.recursive_split(obj, threshold)

        for i, part in enumerate(parts):
            if i == 0:
                continue

            part.parent = obj
            part.matrix_parent_inverse = obj.matrix_world.inverted()

            part.name = f"{original_name}_{i}"


def optimize_scene():
    vertex_limit = 10

    objects_to_process = [o for o in bpy.context.scene.objects if o.type == "MESH"]

    print("--- Starting Mesh Split & Hierarchy Step ---")

    for obj in objects_to_process:
        MeshSplitter.process_object(obj, threshold=vertex_limit)

    all_meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

    print("--- Starting Instance Detection Step ---")

    mesh_registry = {}

    for obj in all_meshes:
        is_none = obj.data is None

        print("Processing Mesh --- " + obj.name + " --- "  + obj.type + " --- " + str(is_none))

        if obj.type != "MESH" or is_none:
            continue

        if len(obj.data.vertices) <= vertex_limit:
            continue

        signature = MeshSignature.get_signature(obj)

        if not signature:
            continue

        if signature in mesh_registry:

            source_obj = mesh_registry[signature]

            if obj.data == source_obj.data:
                continue

            print(f"Match found: {obj.name} is same structure as {source_obj.name}")
            InstanceAligner.align_object(source_obj, obj)

        else:

            mesh_registry[signature] = obj


if __name__ == "__main__":
    print("---------- Running Mesh Optimizer ----------")
    optimize_scene()
