import bmesh
import numpy as np
from mathutils import Vector
import bpy


def create_hyperbolic_paraboloid_mesh(scale_x, scale_y, scale_z, segments):
    mesh = bpy.data.meshes.new(name="HyperbolicParaboloid")
    bm = bmesh.new()

    # Creating vertices
    for i in np.linspace(-1, 1, segments):
        for j in np.linspace(-1, 1, segments):
            x = scale_x * i
            y = scale_y * j
            z = scale_z * i * j  # Hyperbolic paraboloid equation: z = xy
            bm.verts.new((x, y, z))

    bm.verts.ensure_lookup_table()

    # Creating edges and faces
    for i in range(segments - 1):
        for j in range(segments - 1):
            v1 = bm.verts[i * segments + j]
            v2 = bm.verts[i * segments + (j + 1)]
            v3 = bm.verts[(i + 1) * segments + j]
            v4 = bm.verts[(i + 1) * segments + (j + 1)]

            bm.faces.new([v1, v2, v4, v3])

    bm.to_mesh(mesh)
    mesh.update()
    return mesh


def set_shading(object, OnOff=True):
    """ Set the shading mode of an object
        True means turn smooth shading on.
        False means turn smooth shading off.
    """
    if not object:
        return
    if not object.type == 'MESH':
        return
    if not object.data:
        return
    polygons = object.data.polygons
    polygons.foreach_set('use_smooth',  [OnOff] * len(polygons))
    object.data.update()

def toggle_shading(object):
    """ Toggle the shading mode of an object """
    if not object:
        return
    if not object.type == 'MESH':
        return
    if not object.data:
        return
    polygons = object.data.polygons
    for polygon in polygons:
        polygon.use_smooth = not polygon.use_smooth
    object.data.update()

from collections import deque

def approximate_geodesic_distances(mesh, source_index):
    distances = [float('inf')] * len(mesh.vertices)
    distances[source_index] = 0

    queue = deque([source_index])

    while queue:
        current_index = queue.popleft()
        current_distance = distances[current_index]

        for edge in mesh.edges:
            if current_index in edge.vertices[:]:
                next_index = edge.vertices[0] if edge.vertices[1] == current_index else edge.vertices[1]
                edge_length = (mesh.vertices[current_index].co - mesh.vertices[next_index].co).length

                if current_distance + edge_length < distances[next_index]:
                    distances[next_index] = current_distance + edge_length
                    queue.append(next_index)

    return distances

def apply_geodesic_deformation_shape_key(frame_key, distances, max_distance, deformation_strength):
    
    modulus = 11.0  # Adjust this for different modular effects

    
    for i, vertex in enumerate(frame_key.data):
        if distances[i] < max_distance:
            # Deform the vertex based on its distance
            #factor = (1 - distances[i] / max_distance) * deformation_strength
            mod_factor = distances[i] % modulus
            factor = (1 - mod_factor / modulus) * deformation_strength
            
            vertex.co.z += factor  # Example deformation along Z-axis
           # vertex.co.x -= factor*2  # Example deformation along Z-axis
           # vertex.co.y += factor*.5  # Example deformation along Z-axis


def apply_geodesic_deformation(obj, frame_start, frame_end, frame_step, deformation_strength):
    mesh = obj.data
    modulus = 11.0
    # Add a basis shape key (default mesh shape)
    basis_key = obj.shape_key_add(name="Basis", from_mix=False)

    for frame in range(frame_start, frame_end + 1, frame_step):
        bpy.context.scene.frame_set(frame)

        # Add a new shape key for this frame
        frame_key = obj.shape_key_add(name=f"Frame{frame}", from_mix=False)
        frame_key.value = 1.0  # Set the value to 1 to fully apply this shape key

        # Reset vertices to the basis shape before applying new deformation
        for i, v in enumerate(frame_key.data):
            v.co = mesh.vertices[i].co.copy()

        # Apply deformation to the vertices of the shape key
        for v in frame_key.data:
            # Adjust the deformation here; example uses a simple sine wave
            v.co.z += np.sin(v.co.x + frame / 20.0) * deformation_strength #*(frame%modulus)
            v.co.x += np.cos(v.co.y+frame/10) * deformation_strength
            #v.co.y -= np.tan(v.co.z+frame/5) * deformation_strength
            obj.data.update()
        # Insert keyframe for the shape key value
        frame_key.keyframe_insert(data_path="value", frame=frame)

        # Reset other shape keys to 0
        for key in obj.data.shape_keys.key_blocks:
            if key != frame_key:
                key.value = 0
                key.keyframe_insert(data_path="value", frame=frame)


def apply_deformation(obj, frame_start, frame_end, frame_step):
    mesh = obj.data

    # Add a basis shape key (default mesh shape)
    basis_key = obj.shape_key_add(name="Basis", from_mix=False)

    for frame in range(frame_start, frame_end + 1, frame_step):
        bpy.context.scene.frame_set(frame)

        # Add a new shape key for this frame
        frame_key = obj.shape_key_add(name=f"Frame{frame}", from_mix=False)
        frame_key.value = 1.0  # Set the value to 1 to fully apply this shape key

        # Apply deformation to the vertices of the shape key
        for i, v in enumerate(frame_key.data):
            v.co.z += np.sin(v.co.x + frame / 20.0) * 0.1

        # Insert keyframe for the shape key value
        frame_key.keyframe_insert(data_path="value", frame=frame)

        # Reset other shape keys to 0
        for key in obj.data.shape_keys.key_blocks:
            if key != frame_key:
                key.value = 0
                key.keyframe_insert(data_path="value", frame=frame)


import bpy
import bmesh
import numpy as np
from mathutils import Vector

def calculate_mean_curvature(obj):
    # Ensure the object is in the scene and make it the active object
    bpy.context.view_layer.objects.active = obj
    bpy.context.view_layer.update()

    # Switch to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Duplicate the object to not modify the original one
    mesh = obj.data.copy()
    temp_obj = bpy.data.objects.new("TempObject", mesh)
    bpy.context.collection.objects.link(temp_obj)
    bpy.context.view_layer.objects.active = temp_obj
    bpy.context.view_layer.update()

    # Calculate curvature for each vertex
    curvatures = []
    for vertex in temp_obj.data.vertices:
        connected_faces = [face for face in mesh.polygons if vertex.index in face.vertices]
        if len(connected_faces) < 2:
            continue

        normals = [face.normal for face in connected_faces if face.normal.length > 0]
        if len(normals) < 2:
            continue

        angle_sum = 0
        for i in range(len(normals)):
            for j in range(i + 1, len(normals)):
                angle_sum += normals[i].angle(normals[j])

        avg_angle = angle_sum / len(connected_faces)
        curvature = 2 * np.pi - avg_angle
        curvatures.append(curvature)

    # Calculate mean curvature
    mean_curvature = np.mean(curvatures)

    # Normalize to a range of 0 to 1
    normalized_curvature = (mean_curvature - min(curvatures)) / (max(curvatures) - min(curvatures))

    # Clean up
    bpy.data.objects.remove(temp_obj)

    return normalized_curvature

# Other functions follow...



def create_material(obj):
    # Create a new material
    mat = bpy.data.materials.new(name="DynamicMetalMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    while nodes:
        nodes.remove(nodes[0])

    # Create necessary nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    geometry_node = nodes.new(type='ShaderNodeNewGeometry')
    colorramp_node = nodes.new(type='ShaderNodeValToRGB')

    avgcurv = calculate_mean_curvature(obj)

    # Set up the material to be metallic
    bsdf_node.inputs['Metallic'].default_value = 1.0
    bsdf_node.inputs['Roughness'].default_value = .2  # Adjust for desired shininess
    
    avgcurv*=2

    # Set up color ramp to change color based on normal
    colorramp_node.color_ramp.elements[0].color = (avgcurv, 0, 0, 1)  # Red
    colorramp_node.color_ramp.elements[1].color = (0, 0, avgcurv, 1)  # Cyan

    # Link nodes
    links.new(geometry_node.outputs['Normal'], colorramp_node.inputs['Fac'])
    links.new(colorramp_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Set location of the nodes
    output_node.location = (200, 0)
    bsdf_node.location = (0, 0)
    geometry_node.location = (-400, 0)
    colorramp_node.location = (-200, 0)

    return mat


def add_material_to_object(obj):
    mat = create_material(obj)
    # Assign it to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

        
class OBJECT_OT_add_hyperbolic_paraboloid(bpy.types.Operator):
    bl_idname = "mesh.add_hyperbolic_paraboloid"
    bl_label = "Add Hyperbolic Paraboloid"
    bl_options = {'REGISTER', 'UNDO'}

    # ... [previous properties like scale_x, scale_y, scale_z, segments] ...
    scale_x: bpy.props.FloatProperty(
        name="Scale X",
        description="Scale in the X direction",
        default=1.0,
    )

    scale_y: bpy.props.FloatProperty(
        name="Scale Y",
        description="Scale in the Y direction",
        default=1.0,
    )

    scale_z: bpy.props.FloatProperty(
        name="Scale Z",
        description="Scale in the Z direction",
        default=1.0,
    )

    segments: bpy.props.IntProperty(
        name="Segments",
        description="Number of segments",
        min=3, max=100,
        default=24,
    )
    def execute(self, context):
        # ... [previous code in execute method] ...
        mesh = create_hyperbolic_paraboloid_mesh(self.scale_x, self.scale_y, self.scale_z, self.segments)
        obj = bpy.data.objects.new("HyperbolicParaboloid", mesh)

        scene = context.scene
        scene.collection.objects.link(obj)
        obj.location = scene.cursor.location

        # Add material to the object
        add_material_to_object(obj)
        
        # Apply deformation animation
        #apply_deformation(obj, frame_start=1, frame_end=250, frame_step=1)
        # Define deformation strength and frame range
        deformation_strength = 0.081  # Adjust as needed
        start_frame, end_frame, frame_step = 1, 250, 1

        # Apply deformation
        apply_geodesic_deformation(obj, start_frame, end_frame, frame_step, deformation_strength)


    
        return {'FINISHED'}

class OBJECT_PT_hyperbolic_paraboloid(bpy.types.Panel):
    bl_label = "Hyperbolic Paraboloid"
    bl_idname = "OBJECT_PT_hyperbolic_paraboloid"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout

        layout.label(text="Create Hyperbolic Paraboloid")

        row = layout.row()
        row.operator("mesh.add_hyperbolic_paraboloid")

        layout.label(text="Parameters:")
        box = layout.box()
        box.prop(context.scene, 'hp_scale_x')
        box.prop(context.scene, 'hp_scale_y')
        box.prop(context.scene, 'hp_scale_z')
        box.prop(context.scene, 'hp_segments')

def update_hyperbolic_paraboloid(self, context):
    bpy.ops.mesh.add_hyperbolic_paraboloid(
        'INVOKE_DEFAULT', 
        scale_x=context.scene.hp_scale_x,
        scale_y=context.scene.hp_scale_y,
        scale_z=context.scene.hp_scale_z,
        segments=context.scene.hp_segments
    )

def create_parametric_mesh_v2(scale, segments, twist_factor):
    mesh = bpy.data.meshes.new(name="ParametricShapeV2")
    bm = bmesh.new()

    # Creating vertices
    for i in np.linspace(0, 2 * np.pi, segments):
        for j in np.linspace(0, scale, segments):
            r = j
            x = r * np.cos(i)
            y = r * np.sin(i)
            z = 0.5 * r ** 2 * np.sin(2 * i) * twist_factor
            bm.verts.new((x, y, z))

    bm.verts.ensure_lookup_table()

    # Creating faces
    for i in range(segments - 1):
        for j in range(segments - 1):
            v1 = bm.verts[i * segments + j]
            v2 = bm.verts[i * segments + (j + 1)]
            v3 = bm.verts[(i + 1) * segments + j]
            v4 = bm.verts[(i + 1) * segments + (j + 1)]

            bm.faces.new([v1, v2, v4, v3])

    bm.to_mesh(mesh)
    mesh.update()
    return mesh

def create_cayley_cusp_mesh(scale_u, scale_v, segments):
    mesh = bpy.data.meshes.new(name="CayleyCusp")
    bm = bmesh.new()

    # Creating vertices
    du = 2 * scale_u / (segments - 1)
    dv = 2 * scale_v / (segments - 1)
    for i in range(segments):
        for j in range(segments):
            u = -scale_u + i * du
            v = -scale_v + j * dv
            x = u * v
            y = u
            z = v**2
            bm.verts.new((x, y, z))

    bm.verts.ensure_lookup_table()

    # Creating faces
    for i in range(segments - 1):
        for j in range(segments - 1):
            v1 = bm.verts[i * segments + j]
            v2 = bm.verts[i * segments + j + 1]
            v3 = bm.verts[(i + 1) * segments + j + 1]
            v4 = bm.verts[(i + 1) * segments + j]

            # Only create a face if it is not degenerate (all vertices are unique)
            if len({v1, v2, v3, v4}) == 4:
                bm.faces.new([v1, v2, v3, v4])

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()  # Always do this when finished

    return mesh



class OBJECT_OT_add_whitney_umbrella(bpy.types.Operator):
    bl_idname = "mesh.add_whitney_umbrella"
    bl_label = "Add Whitney Umbrella"
    bl_options = {'REGISTER', 'UNDO'}

    scale: bpy.props.FloatProperty(
        name="Scale",
        description="Scale of the shape",
        default=1.0,
    )

    segments: bpy.props.IntProperty(
        name="Segments",
        description="Number of segments",
        min=3, max=100,
        default=24,
    )

    def execute(self, context):
        mesh = create_whitney_umbrella_mesh(self.scale, self.segments)
        obj = bpy.data.objects.new("WhitneyUmbrella", mesh)

        scene = context.scene
        scene.collection.objects.link(obj)
        obj.location = scene.cursor.location

        # Add material to the object
        add_material_to_object(obj)

        return {'FINISHED'}


class OBJECT_OT_add_parametric_shape_v2(bpy.types.Operator):
    bl_idname = "mesh.add_parametric_shape_v2"
    bl_label = "Add Parametric Shape V2"
    bl_options = {'REGISTER', 'UNDO'}

    scale: bpy.props.FloatProperty(
        name="Scale",
        description="Scale of the shape",
        default=1.0,
    )

    segments: bpy.props.IntProperty(
        name="Segments",
        description="Number of segments",
        min=3, max=100,
        default=24,
    )

    twist_factor: bpy.props.FloatProperty(
        name="Twist Factor",
        description="Twist factor of the shape",
        default=0.75,
    )

    def execute(self, context):
        mesh = create_parametric_mesh_v2(self.scale, self.segments, self.twist_factor)
        obj = bpy.data.objects.new("ParametricShapeV2", mesh)

        scene = context.scene
        scene.collection.objects.link(obj)
        obj.location = scene.cursor.location

        # Add material to the object
        add_material_to_object(obj)

        return {'FINISHED'}
    
def create_cayley_cusp_mesh(scale, segments):
    mesh = bpy.data.meshes.new(name="CayleyCusp")
    bm = bmesh.new()

    # Creating vertices
    for s in np.linspace(-scale, scale, segments):
        for t in np.linspace(-scale, scale, segments):
            r = s + 3 * t**2
            x = r
            y = t**3 - r * t
            z = t
            bm.verts.new((x, y, z))

    bm.verts.ensure_lookup_table()

    # Creating faces
    # Faces are created similarly to previous shapes, ensuring they are correctly connected

    bm.to_mesh(mesh)
    mesh.update()
    return mesh

class OBJECT_OT_add_cayley_cusp(bpy.types.Operator):
    bl_idname = "mesh.add_cayley_cusp"
    bl_label = "Add Cayley Cusp"
    bl_options = {'REGISTER', 'UNDO'}

    scale: bpy.props.FloatProperty(
        name="Scale",
        description="Scale of the shape",
        default=1.0,
    )

    segments: bpy.props.IntProperty(
        name="Segments",
        description="Number of segments",
        min=3, max=100,
        default=24,
    )

    def execute(self, context):
        mesh = create_cayley_cusp_mesh(self.scale, self.segments)
        obj = bpy.data.objects.new("CayleyCusp", mesh)

        scene = context.scene
        scene.collection.objects.link(obj)
        obj.location = scene.cursor.location

        # Add material to the object
        add_material_to_object(obj)

        return {'FINISHED'}


def create_branch_point_mesh(scale_r, scale_t, segments):
    mesh = bpy.data.meshes.new(name="BranchPoint")
    bm = bmesh.new()

    # Creating vertices
    for i in range(segments):
        for j in range(segments):
            r = np.linspace(-scale_r, scale_r, segments)[i]
            t = np.linspace(-scale_t, scale_t, segments)[j]
            # Parametric equations for the Branch Point
            x = r * t**2
            y = r  # Assuming u(r,t) = r as in Whitney Umbrella
            z = t
            bm.verts.new((x, y, z))

    bm.verts.ensure_lookup_table()

    # Creating faces
    for i in range(segments - 1):
        for j in range(segments - 1):
            idx = i * segments + j
            # Make sure we don't create a face at the singularity (t=0)
            if j != segments // 2:
                v1 = bm.verts[idx]
                v2 = bm.verts[idx + 1]
                v3 = bm.verts[idx + segments + 1]
                v4 = bm.verts[idx + segments]
                bm.faces.new((v1, v2, v3, v4))

    bm.to_mesh(mesh)
    mesh.update()
    bm.normal_update()

    return mesh

class OBJECT_OT_add_branch_point(bpy.types.Operator):
    bl_idname = "mesh.add_branch_point"
    bl_label = "Add Branch Point"
    bl_options = {'REGISTER', 'UNDO'}

    scale_r: bpy.props.FloatProperty(name="Scale R", default=1.0, description="Scale for the R axis")
    scale_t: bpy.props.FloatProperty(name="Scale T", default=1.0, description="Scale for the T axis")
    segments: bpy.props.IntProperty(name="Segments", default=24, min=3, max=100, description="Number of segments")

    def execute(self, context):
        mesh = create_branch_point_mesh(self.scale_r, self.scale_t, self.segments)
        obj = bpy.data.objects.new("BranchPoint", mesh)

        scene = context.scene
        scene.collection.objects.link(obj)
        obj.location = scene.cursor.location

        # Add material to the object
        add_material_to_object(obj)

        return {'FINISHED'}

import bpy
import bmesh
import numpy as np
import random

# List of potential functions to use in formulas
functions = [
    np.sin,
    np.cos,
    np.tan,
    lambda x: x**2,
    lambda x: x**np.sqrt(abs(x)),
    lambda x: np.sqrt(abs(x)),  # Use abs to avoid complex numbers
    # ... Add more functions as needed
]
# Enhanced Functions List
functions += [
    lambda x: np.exp(x),  # Exponential function
    lambda x: np.log(abs(x) + 1),  # Logarithmic function, avoiding log(0)
    lambda x: np.sinh(x),  # Hyperbolic sine
    # More functions as desired...
]

# List of potential operations to combine functions
operations = [
    #(lambda f, g: lambda u, v: f(u) + g(v)),
    (lambda f, g: lambda u, v: f(u) * g(v)),
    #(lambda f, g: lambda u, v: f(u) - g(v)),
    (lambda f, g: lambda u, v: f(u*g(v)) + g(v*f(u))),
    
    (lambda f, g: lambda u, v: np.conj(f(u)) * g(v)),  # Complex conjugation example

    (lambda f, g: lambda u, v: np.conj(np.sinh(f(u))) * g(v)),
    # ... Add more operations as needed
    #(lambda f, g: lambda u, v: f(np.sinh(u+v*(u+np.abs(v+u))))),
]

def generate_random_formula():
    """Generates a random formula for mesh generation."""
    f = random.choice(functions)
    g = random.choice(functions)
    op = random.choice(operations)
    i = op(f, g)
    j = op(g, f)
    return i 

def create_random_mesh(scale, segments):
    """Creates a random manifold mesh."""
    mesh = bpy.data.meshes.new(name="RandomManifoldMesh")
    bm = bmesh.new()

    formula_x = generate_random_formula()
    formula_y = generate_random_formula()
    formula_z = generate_random_formula()

    # Creating vertices
    for i in np.linspace(-scale, scale, segments):
        for j in np.linspace(-scale, scale, segments):
            x = formula_x(i, j)
            y = formula_y(i, j)
            z = formula_z(i, j)
            bm.verts.new((x, y, z))
         
    bm.verts.ensure_lookup_table()

    # Creating faces
    for i in range(segments - 1):
        for j in range(segments - 1):
            idx = i * segments + j
            v1 = bm.verts[idx]
            v2 = bm.verts[idx + 1]
            v3 = bm.verts[idx + segments + 1]
            v4 = bm.verts[idx + segments]
            try:
                bm.faces.new((v1, v2, v3, v4))
            except ValueError:
                # Avoid creating a face if it already exists
                pass
    
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()  # Free the bmesh to avoid memory leaks

    return mesh

class OBJECT_OT_create_random_mesh(bpy.types.Operator):
    """Operator to create a random manifold mesh."""
    bl_idname = "mesh.create_random_mesh"
    bl_label = "Create Random Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    scale: bpy.props.FloatProperty(name="Scale", default=1.15, min=0.91, max=10.0)
    segments: bpy.props.IntProperty(name="Segments", default=33, min=3, max=100)

    def execute(self, context):
        mesh = create_random_mesh(self.scale, self.segments)
        obj = bpy.data.objects.new("RandomManifoldMesh", mesh)
        scene = context.scene
        scene.collection.objects.link(obj)
        obj.location = scene.cursor.location
        
        toggle_shading(obj)

        #apply_deformation(obj, frame_start=1, frame_end=250, frame_step=1)

         # Add material to the object
        add_material_to_object(obj)
        
        # Apply deformation animation
        #apply_deformation(obj, frame_start=1, frame_end=250, frame_step=1)
        # Define deformation strength and frame range
        deformation_strength = 0.081  # Adjust as needed
        start_frame, end_frame, frame_step = 1, 250, 1

        # Apply deformation
        apply_geodesic_deformation(obj, start_frame, end_frame, frame_step, deformation_strength)


        return {'FINISHED'}



def create_cone_mesh(radius, height, segments):
    mesh = bpy.data.meshes.new(name="Cone")
    bm = bmesh.new()

    # Create the vertex at the tip of the cone
    tip_vertex = bm.verts.new((0, 0, height))

    # Create the vertices for the base of the cone
    base_vertices = [bm.verts.new((radius * np.cos(2 * np.pi * i / segments), 
                                   radius * np.sin(2 * np.pi * i / segments), 0)) for i in range(segments)]

    bm.verts.ensure_lookup_table()

    # Create the base circle edges and side faces
    for i in range(segments):
        next_i = (i + 1) % segments
        # Create the base edge
        bm.edges.new((base_vertices[i], base_vertices[next_i]))
        # Create the side face
        bm.faces.new((base_vertices[i], base_vertices[next_i], tip_vertex))

    # Create the base face
    base_face_verts = bm.verts[-segments:]  # Get the last 'segments' number of vertices
    bm.faces.new(base_face_verts)

    bm.to_mesh(mesh)
    mesh.update()
    return mesh



# Operator to create a cone
class OBJECT_OT_create_cone(bpy.types.Operator):
    bl_idname = "mesh.create_cone"
    bl_label = "Create Cone"
    bl_options = {'REGISTER', 'UNDO'}

    radius: bpy.props.FloatProperty(name="Base Radius", default=1.0, min=0.01, max=100.0)
    height: bpy.props.FloatProperty(name="Height", default=2.0, min=0.01, max=100.0)
    segments: bpy.props.IntProperty(name="Segments", default=32, min=3, max=100)

    def execute(self, context):
        mesh = create_cone_mesh(self.radius, self.height, self.segments)
        obj = bpy.data.objects.new("Cone", mesh)
        scene = context.scene
        scene.collection.objects.link(obj)
        obj.location = scene.cursor.location
        return {'FINISHED'}

# Panel to create a cone
class OBJECT_PT_create_cone(bpy.types.Panel):
    bl_label = "Create Cone"
    bl_idname = "OBJECT_PT_create_cone"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        layout.operator("mesh.create_cone")


# Function to create a Roman Surface
def create_roman_surface(r, segments_theta, segments_phi):
    mesh = bpy.data.meshes.new(name="RomanSurface")
    bm = bmesh.new()

    # Create vertices
    for i in range(segments_theta):
        for j in range(segments_phi):
            theta = np.pi / 2 * (i / (segments_theta - 1))  # theta goes from 0 to pi/2
            phi = 2 * np.pi * (j / (segments_phi - 1))      # phi goes from 0 to 2*pi
            
            # Roman Surface Parametric Equations
            x = r**2 * np.cos(theta) * np.cos(phi) * np.sin(phi)
            y = r**2 * np.sin(theta) * np.cos(phi) * np.sin(phi)
            z = r**2 * np.cos(theta)**2 * np.sin(phi)
            
            bm.verts.new((x, y, z))

    bm.verts.ensure_lookup_table()

    # Create faces
    for i in range(segments_theta - 1):
        for j in range(segments_phi - 1):
            v1 = bm.verts[i * segments_phi + j]
            v2 = bm.verts[i * segments_phi + (j + 1)]
            v3 = bm.verts[(i + 1) * segments_phi + (j + 1)]
            v4 = bm.verts[(i + 1) * segments_phi + j]

            # Create a face only if it's not degenerate
            if v1 != v2 and v2 != v3 and v3 != v4:
                bm.faces.new((v1, v2, v3, v4))

    bm.to_mesh(mesh)
    mesh.update()
    return mesh

# Operator to create a Roman Surface in Blender
class OBJECT_OT_add_roman_surface(bpy.types.Operator):
    bl_idname = "mesh.add_roman_surface"
    bl_label = "Add Roman Surface"
    bl_options = {'REGISTER', 'UNDO'}

    r: bpy.props.FloatProperty(
        name="Radius",
        description="Radius of the Roman Surface",
        default=1.0,
    )
    segments_theta: bpy.props.IntProperty(
        name="Segments Theta",
        description="Number of segments in the theta direction",
        min=3, max=100,
        default=34,
    )
    segments_phi: bpy.props.IntProperty(
        name="Segments Phi",
        description="Number of segments in the phi direction",
        min=3, max=100,
        default=48,
    )

    def execute(self, context):
        mesh = create_roman_surface(self.r, self.segments_theta, self.segments_phi)
        obj = bpy.data.objects.new("RomanSurface", mesh)

        scene = context.scene
        scene.collection.objects.link(obj)
        obj.location = scene.cursor.location

        return {'FINISHED'}

# Panel to create a Roman Surface
class OBJECT_PT_create_roman_surface(bpy.types.Panel):
    bl_label = "Create Roman Surface"
    bl_idname = "OBJECT_PT_create_roman_surface"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        layout.operator("mesh.add_roman_surface")


# Define the transformation function to accept x, y, z coordinates
def transformation(x, y, z):
    # Apply some transformation to the coordinates
    # This is an example, and should be replaced with the desired transformation
    x1= x * y
    y2 = y * z
    z3 = z * x
    
    return x1*x,y2*y,z3*z

    
    
# Define the transformation function to accept x, y, z coordinates
def algo_transformation(x, y, z):
    # Define transformation matrices for different layers
    matrix1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90-degree rotation around Z
    matrix2 = np.array([[np.cos(np.pi/4), 0, -np.sin(np.pi/4)], [0, 1, 0], [np.sin(np.pi/4), 0, np.cos(np.pi/4)]])  # Rotation around Y
    matrix3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])  # Reflection in the YZ plane

    # Apply non-linear transformations
    x = np.sin(x)
    y = np.sinh(y)
    z = np.sin(z)

    # Convert coordinates to a matrix for transformation
    coordinates = np.array([x, y, z])

    # Apply the transformations
    coordinates = np.dot(matrix1, coordinates)
    coordinates = np.dot(matrix2, coordinates)
    coordinates = np.dot(matrix3, coordinates)

    # Introduce additional complexity with non-linear transformations
    x, y, z = coordinates
    x1 = (x) * y
    y2 = y * (y)
    z3 = y * (z)

    return x1, y2, z3
    

# Function to create a transformed parametric surface
def create_transformed_surface(base_function, transform_function, u_range, v_range, segments):
    mesh = bpy.data.meshes.new(name="TransformedParametricSurface")
    bm = bmesh.new()

    # Generate vertices using the base and transformation functions
    for i in range(segments):
        for j in range(segments):
            u = np.linspace(*u_range, segments)[i]
            v = np.linspace(*v_range, segments)[j]
            
            # Apply base function to get initial coordinates (e.g., sphere)
            x, y, z = base_function(u, v)

            # Apply transformation function
            x, y, z = transform_function(x, y, z)
            
            bm.verts.new((x, y, z))

    bm.verts.ensure_lookup_table()

    # Generate faces (similar logic to the Roman surface script)
     # Create faces
    for i in range(segments - 1):
        for j in range(segments - 1):
            v1 = bm.verts[i * segments + j]
            v2 = bm.verts[i * segments + (j + 1)]
            v3 = bm.verts[(i + 1) * segments + (j + 1)]
            v4 = bm.verts[(i + 1) * segments + j]

            # Create a face only if it's not degenerate
            if v1 != v2 and v2 != v3 and v3 != v4:
                bm.faces.new((v1, v2, v3, v4))

    bm.to_mesh(mesh)
    mesh.update()
    return mesh

# Define the base function for a sphere
def sphere_function(u, v, radius):
    radius = 1
    x = radius * np.sin(u) * np.cos(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(u)
    return x, y, z

# Operator to create a transformed parametric surface in Blender
class OBJECT_OT_add_transformed_surface(bpy.types.Operator):
    bl_idname = "mesh.add_transformed_surface"
    bl_label = "Add Transformed Surface"
    bl_options = {'REGISTER', 'UNDO'}

    radius: bpy.props.FloatProperty(name="Radius", default=1.0, min=0.01, max=10.0)
    u_min: bpy.props.FloatProperty(name="U Min", default=0)
    u_max: bpy.props.FloatProperty(name="U Max", default=np.pi)
    v_min: bpy.props.FloatProperty(name="V Min", default=0)
    v_max: bpy.props.FloatProperty(name="V Max", default=2 * np.pi)
    segments: bpy.props.IntProperty(name="Segments", default=32, min=3, max=100)

    def execute(self, context):
        # Create the surface mesh with the transformation applied
        mesh = create_transformed_surface(
            base_function=lambda u, v: sphere_function(u, v, radius=self.radius),
            transform_function=algo_transformation,  # Ensure this function accepts 3 args
            u_range=(self.u_min, self.u_max),
            v_range=(self.v_min, self.v_max),
            segments=self.segments
        )
        obj = bpy.data.objects.new("TransformedParametricSurface", mesh)

        # Add object to the scene
        scene = context.scene
        scene.collection.objects.link(obj)
        obj.location = scene.cursor.location

        return {'FINISHED'}

# Panel to create a transformed parametric surface
class OBJECT_PT_create_transformed_surface(bpy.types.Panel):
    bl_label = "Create Transformed Surface"
    bl_idname = "OBJECT_PT_create_transformed_surface"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        layout.operator("mesh.add_transformed_surface")

from mathutils import Vector
from fractions import Fraction
import fractions
from math import modf
import math

def simplest_fraction_in_interval(x, y):
    """Return the fraction with the lowest denominator in [x,y]."""
    if x == y:
        # The algorithm will not terminate if x and y are equal.
        raise ValueError("Equal arguments.")
    elif x < 0 and y < 0:
        # Handle negative arguments by solving positive case and negating.
        return -simplest_fraction_in_interval(-y, -x)
    elif x <= 0 or y <= 0:
        # One argument is 0, or arguments are on opposite sides of 0, so
        # the simplest fraction in interval is 0 exactly.
        return Fraction(0)
    else:
        # Remainder and Coefficient of continued fractions for x and y.
        xr, xc = modf(1/x);
        yr, yc = modf(1/y);
        if xc < yc:
            return Fraction(1, int(xc) + 1)
        elif yc < xc:
            return Fraction(1, int(yc) + 1)
        else:
            return 1 / (int(xc) + simplest_fraction_in_interval(xr, yr))

def approximate_fraction(x, e):
    """Return the fraction with the lowest denominator that differs
    from x by no more than e."""
    return simplest_fraction_in_interval(x - e, x + e)

# Check if a number is rational and return its denominator in lowest terms
def get_denominator(x, tolerance=.0005):
    try:
        frac = fractions.Fraction(x).limit_denominator()
        if abs(frac - x) < tolerance:
            return frac.denominator
        else:
            return None
    except (TypeError, ValueError):
        return None

# Define a 3D Thomae's function for a point (x, y, z)
def thomae_3d(x, y, z):
    modulus = 7
    tolly = 1.1
    # Get denominators if x, y, and z are rational
    d_x = (math.floor(abs(x*x)*abs(z)))%(math.floor(abs(y*z))+1)
    d_y = (math.floor(abs(y*y)*abs(x)))%(math.floor(abs(z*x))+1)
    d_z = (math.floor(abs(z*z)*abs(y)))%(math.floor(abs(x*y))+1)
    d_x = x*y*y
    d_y = y*z*z
    d_z = z*x*x
    
    if (((math.floor(x*10)%modulus) == 0) and not((math.floor(y*10)%modulus) == 0)):
        return y
    elif ((math.floor(y*10)%modulus) == 0):
        return x
    elif (((math.floor(x*10)%modulus) == 0) and ((math.floor(y*10)%modulus) == 0)):
        return (x-y)*(y-x)
    else:
        return z
    

   # d_x = np.sin(d_x*d_x*d_x*d_x)
   # d_y = np.sin(d_y*d_y*d_y*d_y)
    #d_z = np.sin(d_z*d_z*d_z*d_z)
    
   # if (d_x<d_y and d_y < d_z):
   #     return x*y*z
   # elif (d_y<d_x and d_z>d_y):
     #   return math.floor(x)*math.floor(y)*math.floor(z)-x*y*z
   # elif (d_x<d_y and d_y > d_z):
    #    return math.floor(x)*(y)*(z) 
   # elif (d_y < d_x and d_z < d_x):
    #    return (x)*math.floor(y)*math.floor(z)
   # else:
       # return np.sqrt(((math.ceil(x)*math.floor(y)*math.ceil(z)) +(math.floor(x)*math.ceil(y)*math.floor(z)))/2)*2



# Operator to create a mesh based on 3D Thomae's function
class OBJECT_OT_add_thomae_3d(bpy.types.Operator):
    bl_idname = "mesh.add_thomae_3d"
    bl_label = "Add 3D Thomae Function Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    segments: bpy.props.IntProperty(name="Segments", default=132, min=3, max=200)
    size: bpy.props.FloatProperty(name="Size", default=2.0, min=0.1, max=10.0)

    def execute(self, context):
        mesh = bpy.data.meshes.new(name="Thomae3DMesh")
        bm = bmesh.new()

        for i in range(self.segments):
            for j in range(self.segments):
                 # Map i, j to [-size/2, size/2]
                x = self.size * (i / (self.segments - 1) - 0.5)
                y = self.size * (j / (self.segments - 1) - 0.5)
                z = thomae_3d(x, y, np.sin(x**2 + y**2))  # Apply Thomae's function to the z-coordinate
                
                bm.verts.new((x, y, z))

        bm.verts.ensure_lookup_table()

        # Generate faces
        for i in range(self.segments - 1):
            for j in range(self.segments - 1):
                v1 = bm.verts[i * self.segments + j]
                v2 = bm.verts[i * self.segments + (j + 1)]
                v3 = bm.verts[(i + 1) * self.segments + (j + 1)]
                v4 = bm.verts[(i + 1) * self.segments + j]
                #bm.faces.new((v1, v2, v3, v4))
                
                # Create a face only if it's not degenerate
                if v1 != v2 and v2 != v3 and v3 != v4:
                    bm.faces.new((v1, v2, v3, v4))

        bm.to_mesh(mesh)
        mesh.update()

        # Add mesh as a new object to the scene
        obj = bpy.data.objects.new("Thomae3DMesh", mesh)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location

        return {'FINISHED'}

# Panel to create a Thomae function transformed surface
class OBJECT_PT_add_thomae_3d(bpy.types.Panel):
    bl_label = "Thomae 3D Function"
    bl_idname = "OBJECT_PT_add_thomae_3d"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Create'

    def draw(self, context):
        layout = self.layout
        layout.operator("mesh.add_thomae_3d")

class OBJECT_PT_custom_shapes_v2(bpy.types.Panel):
    bl_label = "Custom Shapes V2"
    bl_idname = "OBJECT_PT_custom_shapes_v2"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout

        layout.label(text="Hyperbolic Paraboloid:")
        box = layout.box()
        box.prop(context.scene, 'hp_scale_x')
        box.prop(context.scene, 'hp_scale_y')
        box.prop(context.scene, 'hp_scale_z')
        box.prop(context.scene, 'hp_segments')
        box.operator("mesh.add_hyperbolic_paraboloid")

        layout.label(text="Parametric Shape V2:")
        box = layout.box()
        box.prop(context.scene, 'ps_scale')
        box.prop(context.scene, 'ps_segments')
        box.prop(context.scene, 'ps_twist_factor')
        box.operator("mesh.add_parametric_shape_v2")
        
        layout.label(text="Whitney Umbrella:")
        box = layout.box()
        box.prop(context.scene, 'wu_scale')
        box.prop(context.scene, 'wu_segments')
        box.operator("mesh.add_whitney_umbrella")
        
        layout.label(text="Cayley Cusp:")
        box = layout.box()
        box.prop(context.scene, 'cc_scale')
        box.prop(context.scene, 'cc_segments')
        box.operator("mesh.add_cayley_cusp")
        
        layout.label(text="Branch Point:")
        box = layout.box()
        box.prop(context.scene, 'bp_scale_r')
        box.prop(context.scene, 'bp_scale_t')
        box.prop(context.scene, 'bp_segments')
        box.operator("mesh.add_branch_point")
        
        layout.label(text="Random Shape:")
        box = layout.box()
        box.operator("mesh.create_random_mesh")
        



def register():
    bpy.utils.register_class(OBJECT_OT_add_hyperbolic_paraboloid)
    bpy.utils.register_class(OBJECT_OT_add_parametric_shape_v2)
    bpy.utils.register_class(OBJECT_PT_custom_shapes_v2)
    bpy.utils.register_class(OBJECT_OT_add_whitney_umbrella)
    bpy.types.Scene.wu_scale = bpy.props.FloatProperty(name="Scale", default=1.0)
    bpy.types.Scene.wu_segments = bpy.props.IntProperty(name="Segments", default=24, min=3, max=100)
    bpy.utils.register_class(OBJECT_OT_add_cayley_cusp)
    bpy.types.Scene.cc_scale = bpy.props.FloatProperty(name="Scale", default=1.0)
    bpy.types.Scene.cc_segments = bpy.props.IntProperty(name="Segments", default=24, min=3, max=100)
    bpy.utils.register_class(OBJECT_OT_add_branch_point)
    bpy.types.Scene.bp_scale_r = bpy.props.FloatProperty(name="Scale R", default=1.0)
    bpy.types.Scene.bp_scale_t = bpy.props.FloatProperty(name="Scale T", default=1.0)
    bpy.types.Scene.bp_segments = bpy.props.IntProperty(name="Segments", default=24, min=3, max=100)
    bpy.utils.register_class(OBJECT_OT_create_random_mesh)
    bpy.utils.register_class(OBJECT_OT_create_cone)
    bpy.utils.register_class(OBJECT_PT_create_cone)
    bpy.utils.register_class(OBJECT_OT_add_roman_surface)
    bpy.utils.register_class(OBJECT_PT_create_roman_surface)
    bpy.utils.register_class(OBJECT_OT_add_transformed_surface)
    bpy.utils.register_class(OBJECT_PT_create_transformed_surface)
    bpy.utils.register_class(OBJECT_OT_add_thomae_3d)
    bpy.utils.register_class(OBJECT_PT_add_thomae_3d)

    bpy.types.Scene.hp_scale_x = bpy.props.FloatProperty(
        name = "Scale X",
        default = 1.0
    )
    bpy.types.Scene.hp_scale_y = bpy.props.FloatProperty(
        name = "Scale Y",
        default = 1.0
    )
    bpy.types.Scene.hp_scale_z = bpy.props.FloatProperty(
        name = "Scale Z",
        default = 1.0
    )
    bpy.types.Scene.hp_segments = bpy.props.IntProperty(
        name = "Segments",
        default = 24,
        min = 3,
        max = 100
    )
    bpy.types.Scene.ps_scale = bpy.props.FloatProperty(
        name = "Scale",
        default = 1.0
    )
    bpy.types.Scene.ps_segments = bpy.props.IntProperty(
        name = "Segments",
        default = 24,
        min = 3,
        max = 100
    )
    bpy.types.Scene.ps_twist_factor = bpy.props.FloatProperty(
        name = "Twist Factor",
        default = 0.5
    )

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_add_hyperbolic_paraboloid)
    bpy.utils.unregister_class(OBJECT_OT_add_parametric_shape_v2)
    bpy.utils.unregister_class(OBJECT_PT_custom_shapes_v2)
    bpy.utils.unregister_class(OBJECT_OT_add_whitney_umbrella)
    bpy.utils.unregister_class(OBJECT_OT_add_cayley_cusp)
    bpy.utils.unregister_class(OBJECT_OT_add_branch_point)
    bpy.utils.unregister_class(OBJECT_OT_create_random_mesh)
    bpy.utils.unregister_class(OBJECT_OT_create_cone)
    bpy.utils.unregister_class(OBJECT_PT_create_cone)
    bpy.utils.unregister_class(OBJECT_OT_add_roman_surface)
    bpy.utils.unregister_class(OBJECT_PT_create_roman_surface)
    bpy.utils.unregister_class(OBJECT_OT_add_transformed_surface)
    bpy.utils.unregister_class(OBJECT_PT_create_transformed_surface)
    bpy.utils.unregister_class(OBJECT_OT_add_thomae_3d)
    bpy.utils.unregister_class(OBJECT_PT_add_thomae_3d)
    
    del bpy.types.Scene.bp_scale_r
    del bpy.types.Scene.bp_scale_t
    del bpy.types.Scene.bp_segments
    del bpy.types.Scene.cc_scale
    del bpy.types.Scene.cc_segments
    del bpy.types.Scene.wu_scale
    del bpy.types.Scene.wu_segments
    del bpy.types.Scene.hp_scale_x
    del bpy.types.Scene.hp_scale_y
    del bpy.types.Scene.hp_scale_z
    del bpy.types.Scene.hp_segments
    del bpy.types.Scene.ps_scale
    del bpy.types.Scene.ps_segments
    del bpy.types.Scene.ps_twist_factor

if __name__ == "__main__":
    register()
