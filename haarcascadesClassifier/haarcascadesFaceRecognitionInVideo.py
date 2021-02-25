import cv2

# Detecting faces in video stream using haarcascades classifier
# Detecting faces in video stream is pretty similar to detecting in images as videos are just a stream
# of images so basically we just have to keep on looping the code over those stream of images and voila
# you have face detection in video

# Loading pre-trained data on face frontals from opencv haarcascades algorithm
trained_face_data = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml')

# To capture video from webcam
# webcam = cv2.VideoCapture(0) # Uncomment this line and comment out line-16 to use your own web cam

# To use a video file as an input
# Uncomment this line and comment out line-13 to use a video
webcam = cv2.VideoCapture('./videos/video.mp4')

# Iterate forever over the frames captured from the video or the webcam
while True:
    # Reading the current frame
    # Here .read() returns two values - a boolean about whether it was able to read the frame and the frame
    succesful_frame_read, frame = webcam.read()
    # if successful_frame_read = false then it means reading frame failed so continuing the loop to retry
    if succesful_frame_read == False:
        continue

    # Converting into grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame in video with faces spotted
    cv2.imshow('Faces', frame)

    # Adding a way to exit the stream/video or to end capturing the video from webcam
    # Listen for a keypress for one millisecond and then move on
    key = cv2.waitKey(1)
    # Stop if 'Q' or 'q' is pressed
    if key == 81 or key == 113:
        break

# Releasing the video capture object
webcam.release()
