import numpy as np
import os
from PIL import Image
import cv2


classmap= {
    0:'car',
    1:'hov',
    2:'person',
    3:'motorcycle'
}
'''
classmap = {0:'pedestrian',
            1:'people',
            2:'bicycle',
            3:'car',
            4:'van',
            5:'truck',
            6:'tricycle',
            7:'awning-tricycle',
            8:'bus',
            9:'motor'}
'''


'''create a xml file'''
out0 = '''<annotation>
    <folder>%(folder)s</folder>
    <filename>%(name)s</filename>
    <path>%(path)s</path>
    <source>
        <database>None</database>
    </source>
    <size>
        <width>%(width)d</width>
        <height>%(height)d</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
'''
out1 = '''    <object>
        <name>%(class)s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%(xmin)d</xmin>
            <ymin>%(ymin)d</ymin>
            <xmax>%(xmax)d</xmax>
            <ymax>%(ymax)d</ymax>
        </bndbox>
    </object>
'''

out2 = '''</annotation>
'''


'''txt to xml'''
def translate(fdir, lists):
    source = {}
    label = {}
    for png in lists:
        print(png)
        if png[-4:] == '.png':
            image = cv2.imread(png)  # 路徑最好不要有中文
            h, w, _ = image.shape

            fxml = png.replace('.png', '.xml')
            fxml = open(fxml, 'w');
            imgfile = png.split('/')[-1]
            source['name'] = imgfile
            source['path'] = png
            source['folder'] = os.path.basename(fdir)

            source['width'] = w
            source['height'] = h

            fxml.write(out0 % source)
            txt = png.replace('.png', '.txt')

            lines = np.loadtxt(txt)  # 讀取txt(已轉成Yolo格式)
            # print(type(lines))

            if len(np.array(lines).shape) == 1:
                lines = [lines]

        for box in lines:
            # print(box.shape)
            if box.shape != (5,):
                box = lines

            # 把txt的類別idx轉成 xml上的類別 (需要在檔案中寫好classsmap)
            label['class'] = classmap[int(box[0])]

            # 把txt的類別idx轉成 xml上的類別idx
            #label['class'] = str(int(box[0]))

            '''Yolo txt格式的xywh是有歸一化後的數字 這邊要轉回xml格式'''
            xmin = float(box[1] - 0.5 * box[3]) * w
            ymin = float(box[2] - 0.5 * box[4]) * h
            xmax = float(xmin + box[3] * w)
            ymax = float(ymin + box[4] * h)

            label['xmin'] = xmin
            label['ymin'] = ymin
            label['xmax'] = xmax
            label['ymax'] = ymax
            fxml.write(out1 % label)
        fxml.write(out2)


if __name__ == '__main__':
    file_dir = "./txt_to_xml"  # 修改
    lists = []
    for i in os.listdir(file_dir):
        if i[-3:] == 'png':   # 注意圖片副檔名
            lists.append(file_dir + '/' + i)
            # print(lists)
    translate(file_dir, lists)
    print('--------txt to xml Done!!!----------')

