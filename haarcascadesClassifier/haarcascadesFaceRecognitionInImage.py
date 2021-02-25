import cv2

# Detecting faces in an image using haarcascades classifier

# Loading pre-trained data on face frontals from opencv haarcascades algorithm
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choosing an image
img = cv2.imread('./images/data.jpg')

# Converting the above image into grayscale image
# We are doing this because haarcascades detects grayscale images as while detecting a
# face color doesn't hold importance as only the facial features like two eyes , nose, mouth, etc
# are used to detect faces .
# So we change the images to grayscale as opencv functions expect a grayscale image as input
# Also note that opencv is smart that is even if you don't convert the image to grayscale and pass it as
# such its not an issue and opencv will do the conversion for you under the hood
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
# .detectMultiScale() detects the desired objects(here faces) of different sizes in the input image
# The detected objects(faces here) are returned as a list of rectangles where a rectangle is
# represented as an array of (x,y,w,h) , here x,y are the coordinates of the top left corner of the face
# and w and h are the width and height of the face
faces = trained_face_data.detectMultiScale(grayscaled_img)
print(faces)

# Outlining the face(coordinates found above) by drawing a rectangle around it
# Syntax - rectangle(img, tupleTopLeftCoordinates, tupleBottomRightCoordinates, colorBGRFormat, rectLineWidth)
# line width parameter is optional
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Visualizing the image with face marked
cv2.imshow('Robert Downey Jr.', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
