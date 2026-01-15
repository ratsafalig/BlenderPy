import bpy
from bpy.types import GizmoGroup, Gizmo

# --- 1. Custom Gizmo Definition (The drawing logic) ---

class AxisLineGizmo(Gizmo):
    """
    A custom gizmo that draws X, Y, and Z axis lines (R, G, B) 
    at the location of the target object.
    """
    bl_idname = "GIZMO_GT_axis_line_gizmo"
    
    # Property to hold the object this specific gizmo is tracking
    target_object: bpy.props.PointerProperty(type=bpy.types.Object)
    
    # Attributes that are not standard bpy properties must be defined in __slots__
    __slots__ = (
        'line_shape_handle', 
        'color_x', 'color_y', 'color_z',
        'axis_length',
    )
    
    def __init__(self):
        # Initialize the custom shape handle to None to prevent AttributeError 
        # if draw() is called before draw_prepare() completes.
        self.line_shape_handle = None

    def draw_prepare(self, context):
        # Only create the custom shape geometry ONCE
        if self.line_shape_handle is not None:
            return 
        
        # --- Define Geometry and Colors ---
        
        self.axis_length = 1.0
        self.line_width = 3

        # Define the vertices for three lines (X, Y, Z)
        verts = [
            (0.0, 0.0, 0.0), (self.axis_length, 0.0, 0.0), # X-Axis
            (0.0, 0.0, 0.0), (0.0, self.axis_length, 0.0), # Y-Axis
            (0.0, 0.0, 0.0), (0.0, 0.0, self.axis_length), # Z-Axis
        ]
        
        # Define the colors (R, G, B) for the vertices
        self.color_x = (1.0, 0.0, 0.0, 1.0) 
        self.color_y = (0.0, 1.0, 0.0, 1.0) 
        self.color_z = (0.0, 0.0, 1.0, 1.0) 

        # Colors must be provided per vertex
        colors = [
            *self.color_x, *self.color_x,
            *self.color_y, *self.color_y,
            *self.color_z, *self.color_z,
        ]

        # Create the custom shape and store the handle
        self.line_shape_handle = self.new_custom_shape('LINES', verts, colors)

    def draw_custom_shape(self, context, shape_handle):
        # This method is automatically called by draw() when using a custom shape handle
        pass

    def draw(self, context):
        # The main draw function simply calls the custom drawing method
        if self.line_shape_handle is not None:
            self.draw_custom_shape(context, self.line_shape_handle)

    def teardown(self):
        # Clean up the custom shape handle when the gizmo is removed 
        # (important for memory management)
        if self.line_shape_handle is not None:
            self.free_custom_shape(self.line_shape_handle)


# --- 2. Gizmo Group Definition (The management logic) ---

class SelectedObjectAxisGizmoGroup(GizmoGroup):
    """
    Manages the creation and destruction of AxisLineGizmos for all selected objects.
    """
    bl_idname = "OBJECT_GGT_selected_axis_gizmos"
    bl_label = "Selected Object Axis Gizmos"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}
    
    # Dictionary to map objects to their active gizmos
    __slots__ = ('gizmo_map',)
    
    @classmethod
    def poll(cls, context):
        # Only activate the gizmo group in Object Mode
        return context.mode == 'OBJECT'

    def setup(self, context):
        # Initialize the map when the group is created
        self.gizmo_map = {}

    def refresh(self, context):
        # Called when the context changes (e.g., selection, frame change)

        # Get current selection and active gizmo objects
        selected_objects = set(context.selected_objects)
        active_gizmo_objects = set(self.gizmo_map.keys())
        
        # Add gizmos for newly selected objects
        for obj in selected_objects - active_gizmo_objects:
            # Check if the object is relevant (e.g., skip cameras/lights if needed)
            if obj.type in {'MESH', 'CURVE', 'EMPTY', 'ARMATURE'}: 
                # Create a new instance of our custom gizmo
                new_gizmo = self.gizmos.new(AxisLineGizmo.bl_idname)
                new_gizmo.target_object = obj
                self.gizmo_map[obj] = new_gizmo
        
        # Remove gizmos for deselected objects
        for obj in active_gizmo_objects - selected_objects:
            gizmo = self.gizmo_map.pop(obj)
            self.gizmos.remove(gizmo)

        # Update the position of all active gizmos
        for obj, gizmo in self.gizmo_map.items():
            # Set the gizmo's basis matrix to the object's world matrix
            # This makes the axes follow the object's position, rotation, and scale
            gizmo.matrix_basis = obj.matrix_world.normalized()


# --- 3. Registration Boilerplate ---

classes = (
    AxisLineGizmo,
    SelectedObjectAxisGizmoGroup,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    # Ensure proper registration and handle reloading if necessary
    try:
        unregister()
    except Exception:
        pass
    register()
    
    print("Custom Axis Gizmos Registered and Active!")