from .util import html


webpage = html.HTML('.','Results Viewer')
for aa in range(500):
    filepaths = ['%s/%s/%i.jpg'%(proc_dir,method_name_out,aa+1) for method_name_out in ['gt']+method_names_out]
    strs = ['gt']+method_names_out
    webpage.add_images(filepaths, strs, filepaths)
    webpage.save(file='index1')