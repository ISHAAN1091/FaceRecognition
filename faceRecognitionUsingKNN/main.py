# Recognise faces using KNN classification algorithm

# 1. Load the training data(numpy arrays of all the persons)
# ----- X - values are stored in numpy arrays
# ----- Y - values we need to assign for each person
# 2. Read a video stream
# 3. Extract faces out of it
# 4. Use KNN to find the prediction of the face
# 5. Map the predicted id to the name of the user
# 6. Display the predictions on the screen - bounding box and name

import numpy as np
import os
import cv2

########## KNN CODE ############
# This is the same KNN algo as we coded earlier in MachineLearningModels/KNN/ repository
# This is just another way of writing that same code . The only major difference here is that
# there we were taking X_train and Y_train separately but here we are taking training data as train
# directly


def distance(v1, v2):
    # Eucledian distance
    return np.sqrt(((v1-v2)**2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
################################


# Initialising camera to capture video from webcam
webcam = cv2.VideoCapture(0)

# Loading pre-trained data on face frontals from opencv haarcascades algorithm
trained_face_data = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml')

# Initialising variables
skip = 0
# Load face images data from images collected using program haarcascadesCollectTrainingData.py
# dataset_path = '../haarcascadesClassifier/data/' # Use this
# Using this for personal reasons , you can use above one without any errors
dataset_path = './data/'
face_data = []  # Used to store all the data of all the files
labels = []  # Used to store labels of all images of all files
class_id = 0  # Used to give unique face value/label for each person's face
names = {}  # Mapping between class_id and names

# Data preparation
# Here we will get all the data of faces from numpy files in /data/ folder and store them in the face_data
# list . Note that here a single numpy file is a collection of images of one person only and is a
# collection of images of that person
# After that we will also create an array of labels storing face values of each image from the numpy file.
# So for a file face values will be same for all images as they are the images of the same person. Also
# note that here face value is just the class_id mapped to that person and this info is stored in names
# dictionary.
# Next we do this for all .npy files we have i.e. for all the people whose faces we have . Also note that we
# keep on increasing the class_id by one in each iteration of for loop as we want each person to have a unique
# face value. Also note that we store the labels array created in label list initialised above
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # Creating a mapping between class_id and name of the person
        names[class_id] = fx[:-4]
        print("Loaded "+fx)
        # Loading and storing data into face_data array
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # Creating labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

# Currently face_data is an array of person's data, where person's data is an array of images, where image is
# a flattened array of pixels . So now we will concatenate face_data to just be just an array of images
# We don't want images of different people to be grouped into an array as well so we are just removing that
# by concatenating
face_dataset = np.concatenate(face_data, axis=0)
# Similarly currently label is an array of person's data's labels, where each person's data's label is an
# array of face value for that person . So now we will concatenate this as well so as to match our face_data
# After concatenating label would be an array of face values
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_dataset.shape, face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

# Testing
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

    for (x, y, w, h) in faces:
        # Extracting Region of Interest (Cropping out the required face)
        # We will also provide the face some padding so as to get a better region of interest
        offset = 10
        face_section = grayscaled_img[y-offset:y+h+offset, x-offset:x+w+offset]
        # Resizing the face section cropped out to get a face image of constant size for each case
        face_section = cv2.resize(face_section, (100, 100))

        # Getting the prediction from KNN
        pred = knn(trainset, face_section.flatten())

        # Display on the screen the name and rectangle around it
        # Getting the respective name of the predicted class_id
        pred_name = names[int(pred)]
        # Displaying the name on the screen
        cv2.putText(frame, pred_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # Displaying the rectangle around the face on the screen
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame and face_section in video with faces spotted
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

####### NOTE #######
# If instead of grayscale images you are having colored images in training data then change
# grayscaled_img to frame in line 127
