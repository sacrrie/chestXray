import cv2
import os
import pyspark as ps
IMPORT='../images'
OUTPUT='../data'

def resize(path,output,size):
    if not os.path.exists(output):
        os.makedirs(output)

    flist=[i for i in os.listdir(path) if i != '.DS_Store']
    for items in flist:
        img=cv2.imread(path+item,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(size,size))
        cv2.imwrite(str(output+item),img)

resize(IMPORT,OUTPUT,size=512)
