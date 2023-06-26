import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# FUNCTION TO SHOW AN IMAGE IN SPYDER
def show(image):
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    # resized = cv2.resize(image, (image.shape[1]//5,image.shape[0]//5))
    resized = cv2.resize(image, (1920,1080))
    # cv2.moveWindow("Output", 200,200)
    cv2.imshow("Output", resized)
    # cv2.resizeWindow("Output", 860, 540)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
THE SECOND PARAMETER IS EITHERR 0 OR 1. 1 MEANS THE SUM IS DONE
TOP TO BOTTOM, 0 MEANS SUM IS DONE HORIZONTALLY. IT RETURNS THE THE ARRAY OF SUM
'''
def Plot_Histogram(image, ax=1):
    x = np.arange(0, img.shape[ax], dtype=np.uint32)
    y = np.zeros(img.shape[ax], dtype=np.uint32)
    for k in range(img.shape[ax]):
        if ax == 1:
            arr = img[:, k]
            y[k] = np.sum(arr)
        else:
            arr = img[k, :]
            y[k] = np.sum(arr)
    plt.plot(x,y)
    plt.show()
    return y


# READING IMAGE, THRESHOLDING AND SHOWING RESULT
path = r"C:\Users\there\Downloads\note.png"
img = cv2.imread(path, 0)
ret3,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img = th3
show(img)
print(ret3)


# FINDING MARGIN START POINT
# print(np.sort(y1))
y = Plot_Histogram(img, 1)
for i in range(img.shape[1]):
    if y[i] < 10000:
        pos = i
        break
print(pos)


# SEPARATING MARGIN REGION
margin_area = img[0:img.shape[0], 0:pos]
show(margin_area)











