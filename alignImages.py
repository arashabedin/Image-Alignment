import cv2 as cv
import numpy as np
from numpy.core.umath_tests import inner1d
from matplotlib import pyplot as plt
from scaleInnerContents import scaleInnerContents
from hausdorffDist import HausdorffDist

def alignImages(img, img2):
    ### Before doing any processing the images should be converted to single channel gray-scale. 
    img_final = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray1 = img_final
    gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    ### We operate or alignment algorithm on an smaller scale of our images and in order to achieve better performance
    img = cv.resize(gray1,(80,60))
    img2 = cv.resize(gray2,(80,60))
    rows,cols = img.shape

    ### We extract the corners of the original image 
    corners2 = cv.goodFeaturesToTrack(img2,100,0.01,10)
    global n2
    n2 = np.squeeze(np.asarray(corners2))

    global bestX , bestY , bestScaleX, bestScaleY, bestScale
    bestX = 0
    bestY = 0
    bestScale = 0
    global smallestDistance
    smallestDistance = 1000

    ### We move the second image in different positions and scale, while comparing the distance between the corners, in each state 
    for i in range(0, 40):
        for j in range(0, 30):
           for k in range(0, 6):
                M = np.float32([[1,0,i-20],[0,1,j-15]])
                dst = cv.warpAffine(img,M,(cols,rows))
                scale = 0.8 + float(( k * 0.1 ))
                dst = scaleInnerContents(dst,scale)

                corners = cv.goodFeaturesToTrack(dst,100,0.01,10)
                n1 = np.squeeze(np.asarray(corners))
                try:
                   distance = HausdorffDist(n1,n2)
                except:
                   distance = 1000
                if distance < smallestDistance:
                    smallestDistance = distance
                    bestX = i
                    bestY = j
                    bestScale = scale


    ### The values we obtained so far, that refers to the best position and scale that our second image should
    ### be transformed to in order to align with the original image
    print(smallestDistance)
    print(bestX)
    print(bestY)
    print(bestScale)
    print("bestScale")
   
    dst = cv.warpAffine(img,M,(cols,rows))
    
    ### Remember that the bestX and bestY properties are calculated according to the smaller version of our images
    ### So we should multipy them by the number which refers to the ratio between the original and smaller size
    bestX = (bestX -20) * 10
    bestY = (bestY -15) * 10

    ### Finaly we align the second image to our orignal image according the parameters of bestX, bestY and bestScale that we
    ### calculated during the operation
    M = np.float32([[1,0,bestX],[0,1,bestY]])
    rows,cols = img_final.shape
    aligned = cv.warpAffine(img_final,M,(cols,rows), borderValue=(255,255,255))
    aligned = scaleInnerContents(aligned,bestScale)


   ### Uncomment the code bellow to see the results
   #  plt.title(' drawing'), plt.xticks([]), plt.yticks([])
   #  plt.imshow(aligned),plt.show()


    return aligned
