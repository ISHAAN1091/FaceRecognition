import cv2
import matplotlib.pyplot as plt

# Reading images
img = cv2.imread('./images/dog.png')
# Above image is read as a 3D matrix of dimensions (num_of_pixels,num_of_pixels,3)
# The 2D image is stored in 3D form storing the RGB value of each pixel
print(img)

# Visualizing the image
# Using matplotlib
# Opencv reads images color codes of its pixels in BGR format rather than the standard RGB
# format used in matplotlib. So changing from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
# Using cv2
# Syntax - imshow(title,already_read_image)
# In this method we don't need to convert from BGR to RGB as imshow method in cv2 uses BGR as default format
cv2.imshow('Dog Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# .waitKey(t) is used to tell the code to close the image window that pops up after t milliseconds
# If you want to keep it on hold and not destroy it automatically you can
# either pass 0 as argument or pass nothing
# .destroyAllWindows() destroys all the windows finally

# Reading images as grayscale images
gray_img = cv2.imread('./images/dog.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Gray Dog Image', gray_img)
cv2.waitKey()
cv2.destroyAllWindows()
