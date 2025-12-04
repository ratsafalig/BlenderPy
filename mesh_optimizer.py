import bpy
import bmesh
import mathutils
from collections import defaultdict
import numpy as np

class MeshUtils:
    """Helper for basic mesh operations."""

    @staticmethod
    def get_vertex_data_world_space(obj):
        """
        Returns vertex coordinates in world space.
        
        Args:
            obj (bpy.types.Object): The Blender object.

        Returns:
            list[mathutils.Vector]: A list of vertex coordinates in world space.
        """
        # 1. Get the object's World Matrix
        world_matrix = obj.matrix_world
        
        # 2. Get the mesh data block
        mesh = obj.data
        
        # 3. Transform local vertex coordinates to world coordinates
        world_verts = []
        for v in mesh.vertices:
            # v.co is the local space coordinate (a Vector)
            # Multiplying the local vector by the 4x4 world matrix transforms it.
            world_coord = world_matrix @ v.co
            world_verts.append(world_coord)
            
        return world_verts

    @staticmethod
    def get_vertex_data(obj):
        """Returns vertex coordinates in local space."""
        # Ensure object is in object mode before accessing mesh data
        # if bpy.context.view_layer.objects.active != obj:
        #     bpy.context.view_layer.objects.active = obj
        # bpy.ops.object.mode_set(mode='OBJECT')
        return [v.co.copy() for v in obj.data.vertices]

    @staticmethod
    def get_center_of_mass(verts):
        """Calculates arithmetic mean of vertices."""
        if not verts:
            return mathutils.Vector((0, 0, 0))
        return sum(verts, mathutils.Vector()) / len(verts)

class MeshSignature:
    """Handles the detection logic for identical meshes."""
    
    @staticmethod
    def calculate_signature(obj):
        """
        Generates a signature tuple: (Vertex Count, Sum of Distances to Center).
        """
        if obj.type != 'MESH' or obj.data is None:
            return None
        
        verts = MeshUtils.get_vertex_data(obj)
        vert_count = len(verts)
        
        if vert_count == 0:
            return (0, 0.0)
            
        center = MeshUtils.get_center_of_mass(verts)
        
        # Algorithm: Sum of distances from center to every vertex
        total_dist = sum((v - center).length for v in verts)
        
        # Rounding to avoid float precision errors
        return (vert_count, round(total_dist, 4))

class InstanceAligner:
    """Handles the transformation matrix calculations based on a canonical basis."""

    @staticmethod
    def get_alignment_basis_legacy(obj):
        """
        Legacy method: Constructs a matrix/plane based on Center, Furthest Point, and Closest Point.
        This method can be unstable for highly symmetrical meshes.
        Returns: (CenterVector, RotationMatrix)
        """
        verts = MeshUtils.get_vertex_data_world_space(obj)
        if len(verts) < 3:
            return None
            
        center = MeshUtils.get_center_of_mass(verts)
        
        # Find Farest and Closest points
        farest_v = max(verts, key=lambda v: (v - center).length)
        closest_v = min(verts, key=lambda v: (v - center).length)
        
        # X-axis: Center -> Farest
        vec_x = (farest_v - center).normalized()
        vec_temp = (closest_v - center).normalized()
        
        # Handle collinear case (rare but possible)
        if abs(vec_x.dot(vec_temp)) > 0.999:
            vec_temp = mathutils.Vector((0, 0, 1)) if abs(vec_x.z) < 0.9 else mathutils.Vector((0, 1, 0))

        # Z-axis: Normal of the plane defined by Center, Far, Close
        vec_z = vec_x.cross(vec_temp).normalized()
        
        # Y-axis: Orthogonal to X and Z
        vec_y = vec_z.cross(vec_x).normalized()
        
        # Create Rotation Matrix from column vectors
        rot_matrix = mathutils.Matrix((vec_x, vec_y, vec_z)).transposed()
        
        return center, rot_matrix
    
    @staticmethod
    def get_alignment_basis_pca(obj):
        """
        Constructs the Rotation Matrix based on Principal Component Analysis (PCA).
        This method is stable and aligns the object to its principal axes of inertia.
        Returns: (CenterVector, RotationMatrix)
        """
        
        # 1. Get Vertex Data and Handle Trivial Case
        verts_blender = MeshUtils.get_vertex_data_world_space(obj)
        N = len(verts_blender)
        
        if N < 3:
            return None
            
        # 2. Convert to NumPy Array and Calculate Centroid
        # PCA requires a NumPy array for efficient matrix operations.
        verts_np = np.array([v.to_tuple() for v in verts_blender])
        
        # Calculate Center of Mass (Centroid)
        center_np = np.mean(verts_np, axis=0)
        center_blender = mathutils.Vector(center_np)
        
        # 3. Center the Data
        P_centered = verts_np - center_np
        
        # 4. Perform SVD to get Principal Axes
        # SVD on the centered data matrix: P_centered = U * S * Vh
        # The rows of Vh (or columns of V) are the principal component directions.
        # We use full_matrices=False for a more compact result.
        U, S, Vh = np.linalg.svd(P_centered, full_matrices=False)
        
        # 5. Extract Principal Directions
        # Vh[0, :], Vh[1, :], Vh[2, :] are the principal axes, sorted by variance (S).
        
        # X-axis: Direction of Maximum Variance (largest eigenvalue/singular value)
        vec_x_np = Vh[0, :]
        # Y-axis: Direction of Second Maximum Variance
        vec_y_np = Vh[1, :]
        # Z-axis: Direction of Minimum Variance (often the plane normal)
        vec_z_np = Vh[2, :]
        
        # --- OPTIONAL: Ensure a Right-Handed Coordinate System ---
        # The SVD decomposition naturally produces an orthonormal basis, 
        # but the orientation might be a reflection (det=-1). 
        # To enforce a right-handed system (like Blender's): Z = X x Y
        
        # Recalculate the Z-axis using the cross product of the first two principal axes.
        # This guarantees the matrix has a determinant of +1 (a pure rotation).
        vec_z_recalc_np = np.cross(vec_x_np, vec_y_np)
        
        # 6. Create Rotation Matrix from Column Vectors
        # Convert NumPy vectors back to Blender vectors
        vec_x = mathutils.Vector(vec_x_np)
        vec_y = mathutils.Vector(vec_y_np)
        vec_z = mathutils.Vector(vec_z_recalc_np) 
        
        # The rotation matrix uses the principal axes as its column vectors:
        # M = [X | Y | Z]
        rot_matrix = mathutils.Matrix([vec_x, vec_y, vec_z]).transposed()
        
        return center_blender, rot_matrix
    
    @staticmethod
    def get_alignment_basis(obj):
        """
        New, more robust method: Calculates the canonical coordinate system (Rotation Matrix)
        using the Principal Component Analysis (PCA) concept via the Inertia Tensor.
        The eigenvectors of the Inertia Tensor provide the principal axes of the 
        geometry's distribution, making it stable for symmetric objects.
        Returns: (CenterVector, RotationMatrix)
        """
        verts = MeshUtils.get_vertex_data_world_space(obj)
        if len(verts) < 3:
            return None
            
        center = MeshUtils.get_center_of_mass(verts)
        verts_centered = [v - center for v in verts]
        
        # Calculate the 3x3 Covariance Matrix (Inertia Tensor) I
        I = mathutils.Matrix.Identity(3)
        
        # I_ij = Sum(v_i * v_j) for centered vertices v
        I[0][0] = sum(v.x * v.x for v in verts_centered)
        I[1][1] = sum(v.y * v.y for v in verts_centered)
        I[2][2] = sum(v.z * v.z for v in verts_centered)
        I[0][1] = I[1][0] = sum(v.x * v.y for v in verts_centered)
        I[0][2] = I[2][0] = sum(v.x * v.z for v in verts_centered)
        I[1][2] = I[2][1] = sum(v.y * v.z for v in verts_centered)

        # The principal axes (rotation) are the eigenvectors of the Covariance Matrix.
        # Since `mathutils` lacks a direct Eigen solver, we extract an orthonormal 
        # basis from the covariance matrix using Gram-Schmidt-like process.
        
        I_cols = [mathutils.Vector(I[i][:3]) for i in range(3)]
        
        # 1. X-axis: Normalized vector of the first column (approximation of max variance axis)
        vec_x = I_cols[0].normalized()
        
        # 2. Y-axis: Orthogonalized projection of the second column
        vec_y = I_cols[1] - (I_cols[1].dot(vec_x) * vec_x)
        
        # Fallback for near-collinear axes
        if vec_y.length < 1e-4:
             # Use a standard world axis that isn't parallel to X
             standard_fallback = mathutils.Vector((0, 0, 1)) if abs(vec_x.z) < 0.9 else mathutils.Vector((0, 1, 0))
             vec_y = standard_fallback - (standard_fallback.dot(vec_x) * vec_x)
             
        vec_y.normalize()
        
        # 3. Z-axis: Cross product of X and Y (guarantees orthogonality)
        vec_z = vec_x.cross(vec_y).normalized()
        
        # Final Rotation Matrix
        rot_matrix = mathutils.Matrix((vec_x, vec_y, vec_z)).transposed()
        
        # Ensure it's a right-handed coordinate system
        if rot_matrix.determinant() < 0:
            rot_matrix[0] = -rot_matrix[0] 
        
        return center, rot_matrix

    @staticmethod
    def align_object(source_obj, target_obj):
        """
        Moves target_obj to match source_obj's visual orientation 
        using the calculated canonical basis (PCA-based), then links the mesh data.
        """
        # Uses the new robust get_alignment_basis
        source_res = InstanceAligner.get_alignment_basis_pca(source_obj)
        target_res = InstanceAligner.get_alignment_basis_pca(target_obj)
        
        if not source_res or not target_res:
            print(f"Skipping alignment for {target_obj.name}: Not enough vertices.")
            return

        s_center, s_rot = source_res
        t_center, t_rot = target_res
        
        # 1. Calculate the rotation difference
        # R = T_rot @ S_rot.inverted()
        rotation_diff = t_rot @ s_rot.inverted()
        
        # 2. Apply to Target Object World Matrix
        # Convert local rotation diff to world space orientation
        target_obj.matrix_world = target_obj.matrix_world @ rotation_diff.to_4x4()
        
        # 3. Swap the Mesh Data (Make it an instance)
        target_obj.data = source_obj.data

class MeshSplitter:
    """Handles splitting objects into smaller chunks."""

    @staticmethod
    def recursive_split(obj, max_verts=100, original_name=None):
        """
        Recursively bisects object until all chunks are under max_verts.
        """
        if original_name is None:
            original_name = obj.name

        # Stop condition
        if len(obj.data.vertices) <= max_verts:
            return [obj]

        # Use Bisection (Plane Cut) through the local center
        # We cut along the longest dimension of the bounding box
        bbox = [mathutils.Vector(b) for b in obj.bound_box]
        center = sum(bbox, mathutils.Vector()) / 8
        dimensions = obj.dimensions
        
        # Determine cut plane normal (X, Y, or Z axis)
        if dimensions.x >= dimensions.y and dimensions.x >= dimensions.z:
            plane_no = (1, 0, 0)
        elif dimensions.y >= dimensions.x and dimensions.y >= dimensions.z:
            plane_no = (0, 1, 0)
        else:
            plane_no = (0, 0, 1)

        # Deselect all, select current object
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        # Enter Edit Mode for Bisection
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        
        # Bisect and Fill
        bpy.ops.mesh.bisect(
            plane_co=center,
            plane_no=plane_no,
            use_fill=True,
            clear_inner=False,
            clear_outer=False
        )
        
        # Separate by loose parts after the cut
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
        
        selected_objects = bpy.context.selected_objects
        
        # If split didn't create new objects
        if len(selected_objects) == 1 and selected_objects[0] == obj:
            print(f"Cannot split solid mesh {obj.name} further with simple separation.")
            return [obj]
            
        # Process the newly created parts recursively
        result_objects = []
        for part in selected_objects:
            result_objects.extend(MeshSplitter.recursive_split(part, max_verts, original_name))
            
        return result_objects

    @staticmethod
    def process_object(obj, threshold=100):
        if obj.type != 'MESH':
            return

        # 1. Check Vertex Count
        if len(obj.data.vertices) <= threshold:
            return

        print(f"Processing Large Object: {obj.name}")
        
        original_name = obj.name
        original_matrix = obj.matrix_world.copy()
        
        # Create a container empty for hierarchy
        container = bpy.data.objects.new(original_name + "_Container", None)
        bpy.context.collection.objects.link(container)
        container.matrix_world = original_matrix
        
        # 2. Loop Cut / Separate: Use recursive_split for actual splitting
        
        # Temporarily store original children/parents before splitting if needed, 
        # but based on the prompt we only care about the newly generated hierarchy.
        
        # Run the split, which leaves the new parts selected
        parts = MeshSplitter.recursive_split(obj, threshold)
        
        # Find the parts that were generated from the original object.
        # This is tricky because the original object might still exist if the split failed 
        # or if the original name was used. We rely on the recursion to handle naming.
        
        # We assume the last 'active' object is the part list from the recursion.
        
        # 3. Hierarchy and Naming
        for i, part in enumerate(parts):
            # Parent to container
            part.parent = container
            # The split parts inherit the original object's location, so we need to zero it out 
            # relative to the container which has the original world matrix.
            part.matrix_parent_inverse = container.matrix_world.inverted()
            
            # Naming
            part.name = f"{original_name}_{i}"
        
        # Delete the original large object if it was successfully split
        # if parts:
        #     bpy.data.objects.remove(obj, do_unlink=True)
            
def optimize_scene():
    vertex_limit = 100
    
    # Snapshot of objects to process (to avoid processing generated ones)
    objects_to_process = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    
    print("--- Starting Mesh Split & Hierarchy Step ---")
    
    # Step 1, 2, 3: Split, Parent, Rename
    for obj in objects_to_process:
        MeshSplitter.process_object(obj, threshold=vertex_limit)
            
    # Refresh list after splitting (includes newly created containers and parts)
    all_meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    
    print("--- Starting Instance Detection Step ---")
    
    # Step 4: Detect Instances
    # Dictionary: signature -> source_object
    mesh_registry = {}
    
    for obj in all_meshes:
        # Only check mesh objects that are actual parts, not containers
        if obj.type != 'MESH' or obj.data is None:
            continue
            
        # We re-evaluate the vertex count check here, although process_object 
        # should have handled objects > limit. This is a safety check.
        if len(obj.data.vertices) <= vertex_limit:
            continue
            
        signature = MeshSignature.calculate_signature(obj)
        
        if not signature:
            continue
            
        if signature in mesh_registry:
            # Step 5: Replace with Variant (Instance)
            source_obj = mesh_registry[signature]
            
            # Check if it's already an instance of the source
            if obj.data == source_obj.data:
                continue
                
            print(f"Match found: {obj.name} is same structure as {source_obj.name}")
            InstanceAligner.align_object(source_obj, obj)
            
        else:
            # Register this as the original/source for this signature
            mesh_registry[signature] = obj

if __name__ == "__main__":
    optimize_scene()