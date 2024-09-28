import numpy as np
import pandas as pd
import cv2

#  Loading Image and Haar Cascade Pre-trained model
## to detect eye and hair from the person's image
img = cv2.imread("sample_image.jpg")           ## Input any image you would like to check
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


#  Getting the position of eye and face
eyes = eye_cascade.detectMultiScale(img, scaleFactor = 1.2, minNeighbors = 4)
faces = face_cascade.detectMultiScale(img, 1.1, 4)


#  Reading the cvs file which contains all the colors
## alongwith its RGB values and initializing some
## global variables
index=["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)
r = g = b = xpos = ypos = 0


#  Function to identify color of the pixel
def recognize_color(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname


#  Function to check the color of the pixel
## by checking it with all the colors we have
## in our csv file
def find_pixel_color(x, y):
    global b, g, xpos, ypos
    xpos = x
    ypos = y
    b, g, r = img[y, x]
    b = int(b)
    g = int(g)
    r = int(r)


cv2.namedWindow('Color Recognition App')


#  Function to get hair color
### I have extrapolated the face's rectangle 25% upward to get the hairs
def check_hair_color():
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y-h//4), (0, 255, 0), 2)
        find_pixel_color(x + w//2, y - h//8)
        text = recognize_color(r, g, b)
        print("Colour of hair comes out to be {}".format(text))


#  Function to get Eyes Color
def check_eyes_color():
    eye = 0
    eye_side = ['left', 'right']
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 0),1)
        find_pixel_color(x + w//2, y + h//2)
        text = recognize_color(r, g, b)
        print("Colour of {} eye comes out to be {}".format(eye_side[eye], text))
        eye = (eye + 1)%2

a = 0
while(1):
    cv2.imshow("Color Recognition App", img)
    if a == 0:
        check = True

    while check:
        check_eyes_color()
        check_hair_color()
        check = False

    a += 1
    
    if cv2.waitKey(20) & 0xFF ==27:
        break
    
cv2.destroyAllWindows()








