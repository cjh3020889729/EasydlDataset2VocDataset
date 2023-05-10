import json
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
import os, sys
import shutil
from tqdm import tqdm
import cv2 as cv

def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def EasyDL2VOC(input_dir, input_json, output_dir, img_name):
    ANNO_SUB_DIR = 'Annotations'
    IMAGE_SUB_DIR = 'JPEGImages'

    img = cv.imread(os.path.join(os.path.join(input_dir, IMAGE_SUB_DIR), img_name))

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = IMAGE_SUB_DIR
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img.shape[1])
    node_height = SubElement(node_size, 'height')
    node_height.text = str(img.shape[0])
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    
    with open(os.path.join(os.path.join(input_dir, 'Annotations'), input_json), 'r') as f:
        json_text = json.load(f)
        labels = json_text.get('labels', None)

        for label in labels:
            name = label.get('name', None)

            x1 = str(label.get('x1', None))
            y1 = str(label.get('y1', None))
            x2 = str(label.get('x2', None))
            y2 = str(label.get('y2', None))

            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = name
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')

            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = x1
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = y1
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = x2
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = y2
    
    indent(node_root)
    ET.ElementTree(node_root).write(os.path.join(os.path.join(output_dir, ANNO_SUB_DIR), input_json.split('.')[0]+'.xml'), xml_declaration=True)
    shutil.copy(os.path.join(os.path.join(input_dir, IMAGE_SUB_DIR), img_name), os.path.join(os.path.join(output_dir, IMAGE_SUB_DIR), img_name))


def extract_dataset(easydl_dataset_dir):
    dir_p = easydl_dataset_dir
    save_dir_p = 'temp_dataset'
    img_dir = os.path.join(save_dir_p, 'JPEGImages')
    anno_dir = os.path.join(save_dir_p, 'Annotations')

    if not os.path.exists(save_dir_p):
        os.mkdir(save_dir_p)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    if not os.path.exists(anno_dir):
        os.mkdir(anno_dir)
    for _, _, files in os.walk(dir_p):
        for f in tqdm(files):
            f_ = os.path.join(dir_p, f)
            if f.endswith('.json'):
                anno_path = os.path.join(anno_dir, f)
                shutil.copy(f_, anno_path)
            else:
                img_path = os.path.join(img_dir, f)
                shutil.copy(f_, img_path)
        break


def create_voc_list(dataset_dir, img_files, train_ratio=0.7):
    if train_ratio >= 1.0:
        train_ratio = 1.0
        print("Warning: The train_ratio should less than or equal to 1.0.")
    if train_ratio <= 0.0:
        print("Error: The train_ratio should more than 0.0.")
        sys.exit(1)
    
    print("Info: Select {0} items.".format(len(img_files)))

    anno_dir = os.path.join(dataset_dir, "Annotations")
    label_list = set()

    _cnt = 0
    _train_size = int(train_ratio*len(img_files))
    with open(os.path.join(dataset_dir, 'label_list.txt'), 'w') as l_f:
        with open(os.path.join(dataset_dir, 'eval_list.txt'), 'w') as e_f:
            with open(os.path.join(dataset_dir, 'train_list.txt'), 'w') as t_f:
                for _img in img_files:
                    _cnt += 1
                    if _cnt >= _train_size:
                        break
                    _anno_path = os.path.join(anno_dir, _img.split('.')[0]+'.xml')
                    tree = ET.parse(_anno_path)
                    root = tree.getroot()
                    objs = root.findall('object')

                    for obj in objs:
                        cls_name = obj.find('name').text
                        label_list.add(cls_name)
                    
                    t_f.write(
                        os.path.join('JEPGImages', _img) + ' ' + \
                        os.path.join('Annotations', _img.split('.')[0]+'.xml') + '\n'
                    )
                print("Info: Select Train Sample: {0}.".format(_cnt))
            for _img in img_files[_cnt:]:
                _anno_path = os.path.join(anno_dir, _img.split('.')[0]+'.xml')
                tree = ET.parse(_anno_path)
                root = tree.getroot()
                objs = root.findall('object')

                for obj in objs:
                    cls_name = obj.find('name').text
                    label_list.add(cls_name)
                
                e_f.write(
                    os.path.join('JEPGImages', _img) + ' ' + \
                    os.path.join('Annotations', _img.split('.')[0]+'.xml') + '\n'
                )
            print("Info: Select Eval Sample: {0}.".format(len(img_files)-_train_size))
        for i in label_list:
            l_f.write(i+'\n')
        print("Info: Select Sample Classes: {0}.".format(label_list))



def generate_from_easydl_to_voc(easydl_dataset_dir, output_dir='voc_dataset', train_ratio=0.85):
    """指定easydl数据集目录, 生成指定VOC格式数据集到输出目录下
        easydl_dataset_dir --> output_dir
            |- Annotations
            |- JPEGImages
            |- label_list.txt
            |- train_list.txt
            |- eval_list.txt
    """
    output_dir = output_dir
    extract_dataset(easydl_dataset_dir)
    input_dir = 'temp_dataset'

    print("\nStart Parse EasyDL Dataset to Voc Dataset!")
    img_files = []
    anno_files = []
    print("\nStart Load Annotations To Parse.")
    for _, _, files in os.walk('temp_dataset/Annotations'):
        anno_files += files
        break
    print("End Load Annotations To Parse.")
    print("\nStart Load JPEGImages To Parse.")
    for _, _, files in os.walk('temp_dataset/JPEGImages'):
        img_files += files
        break
    print("End Load JPEGImages To Parse.")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'Annotations')):
        os.mkdir(os.path.join(output_dir, 'Annotations'))
    if not os.path.exists(os.path.join(output_dir, 'JPEGImages')):
        os.mkdir(os.path.join(output_dir, 'JPEGImages'))

    for _json, _img in tqdm(zip(anno_files, img_files)):
        EasyDL2VOC(input_dir, _json, output_dir, _img)
    print("End Parse EasyDL Dataset to Voc Dataset!")

    print("\nStart Create Voc Dataset List!")
    create_voc_list(output_dir, img_files, train_ratio=train_ratio)
    print("End Create Voc Dataset List!")

