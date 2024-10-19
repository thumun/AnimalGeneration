# imports 
import google.generativeai as genai
#from google.colab import userdata
import os
from os.path import exists, join, basename, splitext
import pandas as pd
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os

######################################################################
# Main 
######################################################################

def main():
    
    #print("Testing image to text gen")

    #gptkey = "AIzaSyBIpE5ZNNQoc-bPs9yB64Ldw0aL_cVbzi8"
    #genai.configure(api_key=gptkey)
    #resultAnimal = callToOpenAI("D:\CV\AnimalGeneration\data\clarissa-cruz-headshot-people-f4197aa2a3b44efb90f907198d950c8d.jpg")
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier("D:\CV\AnimalGeneration\haarcascade_frontalface_default.xml")
    # Read the input image
    img = cv2.imread("D:\CV\AnimalGeneration\gettyimages-1358620879-612x612.jpg")
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    #cv2.imshow('img', img)
    #cv2.waitKey()
    
    # load in training data
    key_pts_frame = pd.read_csv('data/training_frames_keypoints.csv')

    # print out some stats about the data
    print('Number of images: ', key_pts_frame.shape[0])
    
    # a selected image
    n = 120
    image_name = key_pts_frame.iloc[n, 0]
    image = mpimg.imread(os.path.join('data/training/', image_name))
    key_pts = key_pts_frame.iloc[n, 1:].to_numpy()
    key_pts = key_pts.astype('float').reshape(-1, 2)

    print('Image name: ', image_name)

    #plt.figure(figsize=(5, 5))
    #show_keypoints(image, key_pts)
    #plt.show()
         
    
    # Display sunglasses on top of the image in the appropriate place

    # copy of the face image for overlay
    image_copy = np.copy(image)

    # top-left location for sunglasses to go
    # 17 = edge of left eyebrow
    x = int(key_pts[17, 0])
    y = int(key_pts[17, 1])

    # height and width of sunglasses
    # h = length of nose
    h = int(abs(key_pts[27,1] - key_pts[34,1]))
    # w = left to right eyebrow edges
    w = int(abs(key_pts[17,0] - key_pts[26,0]))

    # read in sunglasses
    cat = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)
    #cv2.imshow('img', cat)
    #cv2.waitKey()

    # resize sunglasses
    new_cat =  cv2.resize(cat, (w, h), interpolation = cv2.INTER_CUBIC)

    # get region of interest on the face to change
    roi_color = image_copy[y:y+h,x:x+w]

    # find all non-transparent pts
    ind = np.argwhere(new_cat[:,:,3] > 0)
    print(ind)

    # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
    #print(new_cat.shape)
    for i in range(3):
        #print(new_cat[ind[:,0],ind[:,1],i])
        roi_color[ind[:,0],ind[:,1],i] = new_cat[ind[:,0],ind[:,1],i]/255    
        #plt.show(roi_color)
    # set the area of the image to the changed region with sunglasses
    #print(roi_color.shape)
    #print(roi_color)
    cv2.imshow('img', roi_color)
    cv2.waitKey()
    image_copy[y:y+h,x:x+w] = roi_color


    # display the result!
    #plt.show(image_copy)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', image_copy)
    cv2.waitKey()
    
    pass

#####################################################################
# Extension Functions 
#####################################################################

def apply_filters(face_points, image_copy_1,image_name):

    '''
    Apply animal filters to a person's face

    Parameters:
    --------------------
    face_points: The predicted facial keypoints from the camera
    image_copy_1: Copy of original image

    Returns:
    -------------
    image_copy_1: Animals filters applied to copy of original image
    '''

    animal_filter = cv2.imread("images/"+image_name, cv2.IMREAD_UNCHANGED)

    for i in range(len(face_points)):
        # Get the width of filter depending on left and right eye brow point
        # Adjust the size of the filter slightly above eyebrow points 
        filter_width = 1.1*(face_points[i][14]+15 - face_points[i][18]+15)
        scale_factor = filter_width/animal_filter.shape[1]
        sg = cv2.resize(animal_filter,None, fx=scale_factor, fy = scale_factor, interpolation=cv2.INTER_AREA)
        
        width = sg.shape[1]
        height = sg.shape[0]
        
        # top left corner of animal_filter: x coordinate = average x coordinate of eyes - width/2
        # y coordinate = average y coordinate of eyes - height/2
        x1 = int((face_points[i][2]+5 + face_points[i][0]+5)/2 - width/2)
        x2 = x1 + width

        y1 = int((face_points[i][3]-65 + face_points[i][1]-65)/2 - height/3)
        y2 = y1 + height

        # Create an alpha mask based on the transparency values
        alpha_fil = np.expand_dims(sg[:, :, 3]/255.0, axis=-1)
        alpha_face = 1.0 - alpha_fil
        
        # Take a weighted sum of the image and the animal filter using the alpha values and (1- alpha)
        image_copy_1[y1:y2, x1:x2] = (alpha_fil * sg[:, :, :3] + alpha_face * image_copy_1[y1:y2, x1:x2])
    
    return image_copy_1

def callToOpenAI(img_path):

    myfile = genai.upload_file(img_path)
    print(f"{myfile=}")

    model = genai.GenerativeModel("gemini-1.5-flash")
    result = model.generate_content(
        [myfile, "\n\n", "What animal does this person look like in a one word response?"]
    )
    return (f"{result.text=}")

    pass

def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')

def test():

    pass

if __name__ == "__main__":
    main()