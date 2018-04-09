#REQUIREMENTS
I'm running on Blender 2.79a: https://www.blender.org/download/
You need the VHACD binary, and you also need to install VHACD into blender.

#BLENDER INSTALL DIRECTIONS (REQUIRED)
1. Get VHACD from here: https://github.com/kmammou/v-hacd
2. Copy v-hacd/add-ons/blender/object_vhacd.py to <BLENDER_INSTALL_DIR>/2.79/scripts/addons/
3. Run blender and open the user prefences menu in file dropdown
4. Find Object VHACD option and tick the checkbox
5. Click save user preferences button
6. Set the vhacd_path attribute in visual_mpc/blender_scripts/stl_convert.py to the VHACD binary path
7. If you wish to run VHACD from Blender UI you will need to set VHACD path (directions on github)

#VHACD BUILD DIRECTIONS (OPTIONAL)
The testVHACD binary I pushed was built on Linux Mint with OpenCL disabled.
You may use mine provided it works on your computer. Otherwise use the following steps to build.

1. cd into v-hacd/src/ and run cmake (toggle flags NO_OPENMP and NO_OPENCL as desired)
2. now run make
3. the binary will be in v-hacd/src/testVHACD