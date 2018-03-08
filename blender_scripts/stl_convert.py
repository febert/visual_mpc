########################## run script in blender ###############################
# USAGE
# <BLENDER_PATH>/blender --background --python blender_script.py -- <INPUT_DIR> <OUTPUT_DIR>
################################################################################
import os
from os import path
import sys
import bpy
import glob
############################ constants ############################

vhacd_path=os.path.expanduser("~/visual_mpc/blender_scripts/testVHACD") #set this to testVHACD binary

# https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"
in_path = os.path.expanduser(argv[0])
out_path = os.path.expanduser(argv[1])
print("processing folder", in_path)

# # https://stackoverflow.com/questions/11968976/list-files-in-only-the-current-directory
# files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]

for file_name in glob.glob(in_path +"/*.stl"):
    #reset scene
    scene = bpy.context.scene
    objs = bpy.data.objects
    meshes = bpy.data.meshes
    for obj in objs:
        if obj.type == 'MESH':
            scene.objects.unlink(obj)
            objs.remove(obj)
    for mesh in meshes:
        meshes.remove(mesh)


    output_path = '{}/{}/'.format(out_path, file_name.split('/')[-1].split('.stl')[0])

    try:
        os.makedirs(output_path)
    except OSError:
        if not os.path.isdir(output_path):
            raise

    print("importing stl file:", file_name)
    bpy.ops.import_mesh.stl(filepath=file_name, global_scale=0.1)
    imported = bpy.context.selected_objects[0]
    blender_objname = imported.name

    bpy.data.scenes["Scene"].vhacd.vhacd_path=vhacd_path
    bpy.data.scenes["Scene"].vhacd.data_path = output_path

    bpy.data.objects[blender_objname].select = True
    print("decomposing")
    bpy.ops.object.vhacd(depth = 32, planeDownsampling = 14, convexhullDownsampling = 14)
    print("exporting stl files in batch")
    bpy.ops.export_mesh.stl(filepath=output_path, batch_mode="OBJECT")
    print("finish!")
