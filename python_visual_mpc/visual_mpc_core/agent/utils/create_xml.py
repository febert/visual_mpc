import numpy as np
import xml.etree.cElementTree as ET
import xml.dom.minidom as minidom
import imp
import os
import os

def file_len(fname):
    i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def create_object_xml(hyperparams, load_dict_list=None):
    """
    :param hyperparams:
    :param load_dict_list: if not none load configuration, instead of sampling
    :return: if not loading, save dictionary with static object properties
    """
    xmldir = '/'.join(str.split(hyperparams['filename'], '/')[:-1])
    root = ET.Element("top")

    save_dict_list = []

    for i in range(hyperparams['num_objects']):
        obj = ET.SubElement(root, "body", name="object{}".format(i), pos="0 0 0")
        ET.SubElement(obj, "joint", type="free")

        if load_dict_list == None:
            dict = {}
            color1 = dict['color1'] = np.random.uniform(0.3, 1., 3)
            color2 = dict['color2'] = np.random.uniform(0.3, 1., 3)
            l1 = dict['l1'] =np.random.uniform(0.01, 0.2)
            l2 = dict['l2'] =np.random.uniform(0.01, 0.2)
            save_dict_list.append(dict)
        else:
            dict = load_dict_list[i]
            color1 = dict['color1']
            color2 = dict['color2']
            l1 = dict['l1']
            l2 = dict['l2']

        ET.SubElement(obj, "geom", type="box", size=".03 {} .03".format(l1), rgba="{} {} {} 1".format(color1[0], color1[1], color1[2]))

        pos2 = np.random.uniform(0.01, l1)
        ET.SubElement(obj, "geom", pos="{} {} 0.0".format(l2, pos2), type="box", size="{} .03 .03".format(l2),
                      rgba="{} {} {} 1".format(color2[0], color2[1], color2[2]))

    tree = ET.ElementTree(root)

    xml_str = minidom.parseString(ET.tostring(
        tree.getroot(),
        'utf-8')).toprettyxml(indent="    ")

    xml_str = xml_str.splitlines()[1:]
    xml_str = "\n".join(xml_str)

    with open(xmldir + "/auto_gen/objects{}.xml".format(os.getpid()), "wb") as f:
        f.write(xml_str)

    return save_dict_list


def create_root_xml(hyperparams):
    """
    copy root xml but replace three lines so that they referecne the object xml
    :param hyperparams:
    :return:
    """

    newlines = []
    autoreplace = False
    replace_done =False
    with open(hyperparams['filename']) as f:
        for i, l in enumerate(f):
            if 'begin_auto_replace' in l:
                autoreplace = True
                if not replace_done:
                    newlines.append('        <!--begin auto replaced section -->\n' )
                    newlines.append('        <include file="objects{}.xml"/>\n'.format(os.getpid()))
                    newlines.append('        <!--end auto replaced section -->\n')
                    replace_done = True

            if not autoreplace:
                newlines.append(l)

            if 'end_auto_replace' in l:
                autoreplace = False

    outfile = '/'.join(str.split(hyperparams['filename'], '/')[:-1]) + '/auto_gen/cartgripper{}.xml'.format(os.getpid())
    with open(outfile, 'w') as f:
        for l in newlines:
            f.write(l)

    return outfile

if __name__ == '__main__':
    params = imp.load_source('hyper', "/home/frederik/Documents/catkin_ws/src/visual_mpc/pushing_data/cartgripper_genobj/hyperparams.py")
    agentparams = params.config['agent']
    # create_xml(agentparams)
    create_root_xml(agentparams)