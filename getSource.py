#encoding=utf-8

import os
import cv2

dir_path = './data/source/image'

sseal_path = './data/source/seal'
nnoseal_path = './data/source/noseal'

test_path = './data/source/test'

dev_path = './data/source/dev'

num_seal = 0
num_noseal = 0

noseal_path = os.path.join(dir_path, 'nonseal')
noseal_img_list = os.listdir(noseal_path)
for name in noseal_img_list:
    num = name.split('_')[0]
    # if num != '.XnViewSort' and int(num) > 165000:
    #     continue
    path = os.path.join(noseal_path, name)
    img = cv2.imread(path)
    new_path = os.path.join(nnoseal_path, str(num_noseal) + '.jpg')
    print new_path
    cv2.imwrite(new_path, img)
    num_noseal += 1
seal_path = os.path.join(dir_path, 'seal')
seal_img_list = os.listdir(seal_path)
for name in seal_img_list:
    path = os.path.join(seal_path, name)
    img = cv2.imread(path)
    if num_seal % 10 == 0:
        if num_seal % 20 == 0:
            new_path = os.path.join(test_path, str(num_seal) + '.jpg')
        else:
            new_path = os.path.join(dev_path, str(num_seal) + '.jpg')
    else:
        new_path = os.path.join(sseal_path, str(num_seal) + '.jpg')
    print new_path
    cv2.imwrite(new_path, img)
    num_seal += 1
    if num_seal > 1.2*num_noseal:
        break

