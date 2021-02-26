# Writing a python script that captures images from your webcam video stream
# Extracts all faces from the image frame (using Haarcascades)
# Stores the face information in a numpy array

# 1. Read and show video stream , capture images
# 2. Detect faces and show bounding box
# 3. For every 10th image flatten the largest face image array and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

#######
# If while running this script you are getting a name error: face_section not defined
# Then that means you are sitting in a dark area and your face not

import cv2
import numpy as np

# Initialising camera to capture video from webcam
webcam = cv2.VideoCapture(0)

# Loading pre-trained data on face frontals from opencv haarcascades algorithm
trained_face_data = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml')

# Initialising variables
skip = 0
face_data = []
dataset_path = './data/'
file_name = input('Enter name of the person: ')

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

    # Sorting the faces received from above to find out the largest face
    # The largest face would have the largest area meaning for that face in faces
    # face[2]*face[3] would be maximum
    # So sorting faces according to the above factor
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)

    # Draw rectangles around the faces and cropping out the face for storing as training data
    if len(faces) != 0:
        # Using only the first face from faces list as we need only the largest face
        (x, y, w, h) = faces[0]
        # Drawing rectangles
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extracting Region of Interest (Cropping out the required face)
        # We will also provide the face some padding so as to get a better region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        # Resizing the face section cropped out to get a face image of constant size for each case
        face_section = cv2.resize(face_section, (100, 100))

        # Storing largest face every 10th image
        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))
        skip += 1
    else:
        # Assigning a black grayscale image to face_section as in the case the surroundings are dark or
        # the camera quality is poor then openCV won't identify any face and faces list would be empty
        # due to which the for loop won't be executed so we need a placeholder value which can be displayed
        # by the imshow method in such a case that is face_section placeholder value should also be an image
        # and not a empty list or None otherwise the program throws an error as in that case datatype required
        # by the imshow method and the datatype of face_section passed won't match
        face_section = np.zeros((28, 28))

    # Display the frame and face_section in video with faces spotted
    cv2.imshow('Faces', frame)
    cv2.imshow('Face section', face_section)

    # Adding a way to exit the stream/video or to end capturing the video from webcam
    # Listen for a keypress for one millisecond and then move on
    key = cv2.waitKey(1)
    # Stop if 'Q' or 'q' is pressed
    if key == 81 or key == 113:
        break

# Converting our face data array(4D) containing list of faces(Each face image is 3D array) into a numpy array
# and flattening all the images to make face_data a 2D array
face_data = np.asarray(face_data)
print(face_data.shape)  # Here face_data is 4D array
# Flattening face_section images
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)  # Here face_data is 2D array

# Saving this data file into system
np.save(dataset_path+file_name+'.npy', face_data)
print('Data successfully saved at '+dataset_path+file_name+'.npy')

# Releasing the video capture object
webcam.release()
cv2.destroyAllWindows()
