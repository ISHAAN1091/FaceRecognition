import cv2
import numpy as np
import pandas as pd


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


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('Nose18x15.xml')

img = cv2.imread('./images/Before.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

glasses = cv2.imread('./images/glasses.png', -1)
mustache = cv2.imread('./images/mustache.png', -1)

eye = eye_cascade.detectMultiScale(gray, 1.1, 5)
for x, y, w, h in eye:
    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    glasses = cv2.resize(glasses, (w, h))
    overlay_image_alpha(img, glasses[:, :, 0:3],
                        (x, y), glasses[:, :, 3]/255.0)

nose = nose_cascade.detectMultiScale(gray, 1.1, 5)
for x, y, w, h in nose:
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    mustache = cv2.resize(mustache, (w, h))
    h = int(h/2)
    w = int(w/2)
    overlay_image_alpha(
        img, mustache[:, :, 0:3], (x, y+h), mustache[:, :, 3]/255.0)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert into csv to store the image
prediction = np.array(img)
prediction = prediction.reshape((-1, 3))
print(prediction.shape)

df = pd.DataFrame(data=prediction, columns=[
                  'Channel 1', 'Channel 2', 'Channel 3'])
df.to_csv('result.csv', index=False)
