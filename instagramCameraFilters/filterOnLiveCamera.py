import cv2
import numpy as np

# Applying face filters in realtime video


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (
            alpha * img_overlay[y1o:y2o, x1o:x2o, c] + alpha_inv * img[y1:y2, x1:x2, c])


# Loading pre-trained data on face frontals from opencv haarcascades algorithm
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('Nose18x15.xml')


# To use a video file as an input
# Uncomment this line and comment out line-13 to use a video
webcam = cv2.VideoCapture(0)

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

    # Loading images to apply in face filters
    glasses = cv2.imread('./images/glasses.png', -1)
    mustache = cv2.imread('./images/mustache.png', -1)

    eye = eye_cascade.detectMultiScale(grayscaled_img, 1.1, 5)
    for x, y, w, h in eye:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        glasses = cv2.resize(glasses, (w, h))
        overlay_image_alpha(frame, glasses[:, :, 0:3],
                            (x, y), glasses[:, :, 3]/255.0)

    nose = nose_cascade.detectMultiScale(grayscaled_img, 1.1, 5)
    for x, y, w, h in nose:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        mustache = cv2.resize(mustache, (w, h))
        h = int(h/2)
        w = int(w/2)
        overlay_image_alpha(
            frame, mustache[:, :, 0:3], (x, y+h), mustache[:, :, 3]/255.0)

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
cv2.destroyAllWindows()
