import cv2
import numpy as np
import csv
import os

for image in os.listdir('./img'):
    #print(image)
    img_path = './img/' + image
    #print(img_path)
    img0 = cv2.imread(img_path)
    img_h, img_w, _ = img0.shape
    # 得到這張圖片的H W
    #print(image, img_h, img_w)
    
    img = image.split('.')[0]
    txt_path = './label/' + img + '.txt'
    new_txt_path = './new_label/' + img + '.txt'
    #print(txt_path)
    
    with open(new_txt_path, 'w+', newline='') as csvfile:
        # 以空白分隔欄位，建立 CSV 檔寫入器
        writer = csv.writer(csvfile, delimiter=' ') # 空格分隔
        
        f = open(txt_path, "r")
        lines = f.readlines() # lines是一個大list
        for i in lines:            
            line = i.split('\n')[0]
            cls, LTx, LTy, w, h = line.split(',')
            
            x_c = (int(LTx) + int(w)/2)/img_w    # 歸一化後的 中心點x
            y_c = (int(LTy) + int(h)/2)/img_h    # 歸一化後的 中心點y
            w_end = int(w)/img_w  # 歸一化後的 W
            h_end = int(h)/img_h  # 歸一化後的 H
            
            ''' 自己決定要取到小數點後底幾位 不做的話也可以但會拖很長
            x_c = np.round(x_c, 10)
            y_c = np.round(y_c, 10)
            w_end = np.round(w_end, 10)
            h_end = np.round(h_end, 10) # 小數點取到哪一位自己調
            '''                        
            writer.writerow([cls, x_c, y_c, w_end, h_end]) 