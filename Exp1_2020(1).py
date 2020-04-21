"""
Task [I] - Demonstrating how to compute the histogram of an image using 4 methods.
(1). numpy based
(2). matplotlib based
(3). opencv based
(4). do it myself (DIY)
check the precision, the time-consuming of these four methods and print the result.
"""

import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


img = cv2.imread('C:/Users/Feimao/Desktop/test.jpg')
# plt.hist(img.ravel(), bins=256, range=[0, 256]);
# plt.show()

# blurred = np.hstack([cv2.GaussianBlur(img,(3,3),0),
#                      cv2.GaussianBlur(img,(5,5),0),
#                      cv2.GaussianBlur(img,(7,7),0)
#                      ])
# cv2.imshow("Gaussian",blurred)
# cv2.waitKey(0)
img[:, :, 1] = ndimage.gaussian_filter(img[:, :, 1], 3)
green_img=img[:,:,1]
green_img = ndimage.gaussian_filter(green_img, 40)
cv2.imshow('after',green_img)

plt.hist(img[:, :, 1].ravel(), bins=250, color='r')
plt.show()



cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


###
#please coding here for solving Task [I].









###





"""
Task [II]Refer to the link below to do the gaussian filtering on the input image.
Observe the effect of different @sigma on filtering the same image.
Try to figure out the gaussian kernel which the ndimage has used [Solution to this trial wins bonus].
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
"""

from scipy import ndimage
img = cv2.imread('C:/Users/Feimao/Desktop/test.jpg')

index = 150
plt.subplot(index)
plt.imshow(img)
for sigma in (2, 5, 10):
    im_blur = np.zeros(img.shape, dtype=np.uint8)
    for i in range(3):
        im_blur[:, :, i] = ndimage.gaussian_filter(img[:, :, i], sigma)
    index += 1
    plt.subplot(index)
    plt.imshow(im_blur)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()


###
#please coding here for solving Task[II]









"""
Task [III]  Check the following link to accomplish the generating of random images.
Measure the histogram of the generated image and compare it to the according gaussian curve
in the same figure.
"""

mean = (2, 2,2)
cov = np.eye(3)
x = np.random.multivariate_normal(mean, cov, (600, 600))
plt.hist(x.ravel(), bins=200, color='r')
plt.show()

plt.hist(x.ravel(), bins=128, color='r')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
###
#please coding here for solving Task[III]



