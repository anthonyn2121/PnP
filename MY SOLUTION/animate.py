import cv2 
import numpy as np 
import glob 


img_array = []
for filename in sorted(glob.glob('my_results/*.png')):
    img = cv2.imread(filename)
    height,width,layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('motion.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()