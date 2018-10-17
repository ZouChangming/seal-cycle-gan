#encoding=utf-8

import os
import cv2
try:
  import xml.etree.cElementTree as ET
except ImportError:
  import xml.etree.ElementTree as ET

seal_path = './data/source/seal'
noseal_path = './data/source/noseal'

path = './data/xml'

img_xml_list = os.listdir(path)
img_xml_list.sort()

c_s = 0
c_n = 0

for i in range(len(img_xml_list)/2):
    print os.path.join(path, img_xml_list[i*2])
    print os.path.join(path, img_xml_list[i*2+1])
    img = cv2.imread(os.path.join(path, img_xml_list[i*2]))
    tree = ET.parse(os.path.join(path, img_xml_list[i*2+1]))
    root = tree.getroot()
    for Object in root.findall('Block'):
        for line in Object.findall('Line'):
            pts = line.attrib['Points'].split(';')
            p1 = pts[0].split(',')
            p2 = pts[2].split(',')
            label = line.attrib['Label']
            crop_img = img[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0])]
            if label == '1':
                cv2.imwrite(os.path.join(seal_path, str(c_s) + '.jpg'), crop_img)
                c_s += 1
            elif label == '0':
                cv2.imwrite(os.path.join(noseal_path, str(c_n) + '.jpg'), crop_img)
                c_n += 1
