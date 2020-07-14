# coding: utf-8
import argparse
import xml.etree.ElementTree as ET
import os
import sys

""" lib """
from object_detection.utils import label_map_util
from lib.utility import load_config

def parse_args():
  parser = argparse.ArgumentParser(description='generate val txt')
  parser.add_argument(
    "--config_path",\
    type = str,\
    default="/tf/minda/github/detect_ws/cfg/data/parse_voc_xml.json",\
    help="config path")

  args = parser.parse_args()
  return args

def create_category_name(categories):
  category_index = {}
  for cat in categories:
    category_index[cat['name']] = cat['id']
  return category_index

def parse_xml(path, category_name):
  tree = ET.parse(path)
  img_name = path.split('/')[-1][:-4]
    
  height = tree.findtext("./size/height")
  width = tree.findtext("./size/width")

  objects = [img_name, width, height]

  for obj in tree.findall('object'):
    difficult = obj.find('difficult').text
    if difficult == '1':
      continue
    
    name = obj.find('name').text
    bbox = obj.find('bndbox')
    xmin = bbox.find('xmin').text
    ymin = bbox.find('ymin').text
    xmax = bbox.find('xmax').text
    ymax = bbox.find('ymax').text

    name = str(category_name[name])
    objects.extend([name, xmin, ymin, xmax, ymax])
  
  if len(objects) > 1:
    return objects
  else:
    return None

def gen_test_txt(cfg, category_name):
  test_path = [os.path.join(cfg['PATH_TO_DATASET'], 'ImageSets/Main/test.txt')]
  anno_path = [os.path.join(cfg['PATH_TO_DATASET'], 'Annotations')]
  img_path = [os.path.join(cfg['PATH_TO_DATASET'], 'JPEGImages')]

  with open(cfg['OUT_PATH'], "w") as outfile:
    test_cnt = 0
    for i, path in enumerate(test_path):
      img_names = open(path, 'r').readlines()
      for img_name in img_names:
        img_name = img_name.strip()
        xml_path = anno_path[i] + '/' + img_name + '.xml'
        objects = parse_xml(xml_path, category_name)
        
        if objects:
          objects[0] = img_path[i] + '/' + img_name + '.jpg'
          if os.path.exists(objects[0]):
            objects.insert(0, str(test_cnt))
            test_cnt += 1
            objects = ' '.join(objects) + '\n'
            outfile.write(objects)
    
    print("out_path : {}".format(cfg['OUT_PATH']))
    print("success")

def main():
  args = parse_args()
  cfg = load_config.readCfg(args.config_path)
  
  """ label """
  label_map = label_map_util.load_labelmap(cfg["PATH_TO_LABELS"])
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes=cfg['NUM_CLASSES'], use_display_name=True)
  category_name = create_category_name(categories)
  
  print("dataset class name")
  print(category_name)
  
  gen_test_txt(cfg, category_name)
  
    
if __name__ == "__main__":
  main()


