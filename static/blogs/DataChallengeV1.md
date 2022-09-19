# Subject Identification with Computer Vision using Python

The following code creates an algorithm that identifies the subject of an image, outlines it, and crops the image to isolate the subject. 

By Shivant Maharaj


```python
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
```


```python
newpath = 'C:\DataScienceChallengeOutput'
cnt = 0

if not os.path.exists(newpath):
    os.makedirs(newpath)
```


```python
#Method to process images
def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 12, 93)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=10)
    img_erode = cv2.erode(img_dilate, kernel, iterations=10)
    return img_erode

#Method to scale down size of images
def scaleDown(origImg):
    #percent by which the image is resized
    scale_percent = 50

    #calculate the 50 percent of original dimensions
    width = int(origImg.shape[1] * scale_percent / 100)
    height = int(origImg.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(origImg, dsize)
    return output


#Main Method
for x in range(1, 15):
    
    cnt = cnt + 1
    
    #Load Image
    imgName = 'Image' + str(cnt) + '.jpg'
    
    #Creating Variables
    src = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
    srcCopy = cv2.imread(imgName, cv2.IMREAD_UNCHANGED) #because the mask was blue and method overwrote the smallImg file
    smallImg = scaleDown(src)
    rectImg = smallImg 
    copyImg = scaleDown(srcCopy) 
    
    #Finding contours of foreground image
    contours,_ = cv2.findContours(process(smallImg), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contourImg = cv2.drawContours(smallImg, contours, -1, (0, 255, 0), 2)
    
    #Finding max foreground image and plotting rectangle
    c = max(contours,key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(rectImg, (x,y), (x+w,y+h), (255,0,0),2)
    
    #Cropping image with coordinates identified by Max Contour
    croppedImg = copyImg[y:y+h, x:x+w]
    
    #Saving Images 
    cv2.imwrite(('C:\DataScienceChallengeOutput\OriginalImage' + str(cnt) + '.jpg'),copyImg)
    cv2.imwrite(('C:\DataScienceChallengeOutput\ContouredImage' + str(cnt) + '.jpg'),rectImg)
    cv2.imwrite(('C:\DataScienceChallengeOutput\FinalImage' + str(cnt) + '.jpg'), croppedImg)
    
    #Testing Output 
    #cv2.imshow("OrigImage", copyImg)
    #plt.imshow(cv2.cvtColor(copyImg, cv2.COLOR_BGR2RGB))
    
    #cv2.imshow("RectImage", rectImg)
    #plt.imshow(cv2.cvtColor(rectImg, cv2.COLOR_BGR2RGB))
    
    #cv2.imshow("croppedImage", croppedImg)
    #plt.imshow(cv2.cvtColor(croppedImg, cv2.COLOR_BGR2RGB))
    
    #cv2.waitKey(0)
    

   
    
```


```python
  
```


```python

```


```python

```


```python

```
