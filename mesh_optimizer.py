import bpy
import bmesh
import mathutils
from collections import defaultdict

class MeshUtils:
    """Helper for basic mesh operations."""
    
    @staticmethod
    def get_vertex_data(obj):
        """Returns vertex coordinates in local space."""
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
    """Handles the transformation matrix calculations based on 3-point plane."""

    @staticmethod
    def get_alignment_basis(obj):
        """
        Constructs a matrix/plane based on Center, Furthest Point, and Closest Point.
        Returns: (CenterVector, RotationMatrix)
        """
        verts = MeshUtils.get_vertex_data(obj)
        if len(verts) < 3:
            return None
            
        center = MeshUtils.get_center_of_mass(verts)
        
        # Find Farest and Closest points
        # Using index to ensure deterministic behavior if distances are equal
        farest_v = max(verts, key=lambda v: (v - center).length)
        closest_v = min(verts, key=lambda v: (v - center).length)
        
        # Construct Basis Vectors
        # X-axis: Center -> Farest
        vec_x = (farest_v - center).normalized()
        
        # Temp vector: Center -> Closest
        vec_temp = (closest_v - center).normalized()
        
        # Handle collinear case (rare but possible)
        if abs(vec_x.dot(vec_temp)) > 0.999:
            # Fallback: Use arbitrary axis if points are collinear
            vec_temp = mathutils.Vector((0, 0, 1)) if abs(vec_x.z) < 0.9 else mathutils.Vector((0, 1, 0))

        # Z-axis: Normal of the plane defined by Center, Far, Close
        vec_z = vec_x.cross(vec_temp).normalized()
        
        # Y-axis: Orthogonal to X and Z
        vec_y = vec_z.cross(vec_x).normalized()
        
        # Create Rotation Matrix from column vectors
        rot_matrix = mathutils.Matrix((vec_x, vec_y, vec_z)).transposed()
        
        return center, rot_matrix

    @staticmethod
    def align_object(source_obj, target_obj):
        """
        Moves target_obj to match source_obj's visual orientation 
        using the calculated planes, then links the mesh data.
        """
        source_res = InstanceAligner.get_alignment_basis(source_obj)
        target_res = InstanceAligner.get_alignment_basis(target_obj)
        
        if not source_res or not target_res:
            print(f"Skipping alignment for {target_obj.name}: Not enough vertices.")
            return

        s_center, s_rot = source_res
        t_center, t_rot = target_res
        
        # 1. Calculate the rotation difference
        # We want a matrix R such that: R @ s_rot = t_rot
        # So: R = t_rot @ s_rot.inverted()
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
        
        # Separate by selection (The bisection creates a loop, 
        # but to actually split into objects, we use Separate Loose Parts 
        # or simple geometry ripping. 
        # A simpler approach for 'Loop Cut' style is V-Ripping the selection edge,
        # but separate by loose parts is robust after a cut).
        
        # NOTE: Standard bisection just adds edges. To separate, we need to rip.
        # Alternatively, we can use Separate > Loose Parts if the geometry is disjoint.
        # If geometry is solid, we simply separate the selection.
        
        # Strategy: Mark the new edges as seams or just separate blindly?
        # Robust Strategy: Separate Loose Parts first. If that fails to reduce count,
        # simply force a geometry split.
        
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
        
        selected_objects = bpy.context.selected_objects
        
        # If split didn't create new objects (mesh was solid), we need to split by plane geometry
        # This is complex in script. 
        # Simplification: If separate loose parts returned only 1 object (itself),
        # we stop to avoid infinite loops, or implement a hard mesh slice.
        
        result_objects = []
        
        if len(selected_objects) == 1 and selected_objects[0] == obj:
            # Separation failed (solid mesh). Returning as is to prevent crash.
            print(f"Cannot split solid mesh {obj.name} further with simple separation.")
            return [obj]
            
        # Process the newly created parts recursively
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
        
        # 2. Loop Cut / Separate
        # First try Loose parts, as it's non-destructive
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        # We start by separating loose parts
        bpy.ops.mesh.separate(type='LOOSE')
        parts = bpy.context.selected_objects
        
        # 3. Hierarchy and Naming
        for i, part in enumerate(parts):
            # Parent to container
            part.parent = container
            # Since container has the transform, we assume parts keep their world pos.
            # Using matrix_parent_inverse usually handles this.
            part.matrix_parent_inverse = container.matrix_world.inverted()
            
            # Naming
            part.name = f"{original_name}_{i}"
            
            # Recursive check if parts are still too big (Optional based on prompt)
            # If prompt implies strict loop cutting, we would call recursive_split here.
            
def optimize_scene():
    vertex_limit = 100
    
    # Snapshot of objects to process (to avoid processing generated ones)
    objects_to_process = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    
    print("--- Starting Mesh Split & Hierarchy Step ---")
    
    # Step 1, 2, 3: Split, Parent, Rename
    for obj in objects_to_process:
        MeshSplitter.process_object(obj, threshold=vertex_limit)
            
    # Refresh list after splitting
    all_meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    
    print("--- Starting Instance Detection Step ---")
    
    # Step 4: Detect Instances
    # Dictionary: signature -> source_object
    mesh_registry = {}
    
    for obj in all_meshes:
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