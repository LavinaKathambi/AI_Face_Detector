
# Import the cv2 library; importing directly "import cv2" might briing an error, so you can type this 
from cv2 import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv(haar cascade algorithm)
# Assign this to a variable and call the opencv library; the CascadeClassifier function to make a classifier
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect the faces in; call the image read (imread) function from opencv
img = cv2.imread('pic.jpg')

# Convert the image to greyscale; Call the convert color function; img and color are the paramaters
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Detect faces by calling detectMultiScale function; Detects objects of different sizes in the input image; returns as a list of rectangles
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


# Draw the Rectangles, pick a color, BGR (opencv is backwards), choose the intensity
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)

# Show the image using the image show (imshow) function from opencv
cv2.imshow('Human Face Detector', img)

# Call the waitKey function to prevent the image from closing instantly; this pauses the execution of your code; press any key to continue
key = cv2.waitKey()

print("Complete!")
