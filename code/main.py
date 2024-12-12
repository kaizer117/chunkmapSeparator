import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('./resources/example.png', cv.IMREAD_UNCHANGED)

# #make mask of where the transparent bits are
# trans_mask = img[:,:,3] == 0

# #replace areas of transparency with white and not transparent
# img[trans_mask] = [255, 255, 255, 255]

# cv.imshow("Display window", img)
# k = cv.waitKey(0)

plt.imshow(img)
plt.show()
print("success")