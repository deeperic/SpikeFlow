#i need this to generate chinese character images

import cv2
import os
import sys


def createFolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
        
def sort_bounding_boxes(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = cnts
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
	
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
	
def captch_ex(file_name, folder1, folder2 ):
    img  = cv2.imread(file_name)
    img2 = cv2.imread(file_name) #for saving
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    new_img = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 10)
    need_img = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 15)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3 , 3)) # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more 
    dilated = cv2.dilate(new_img,kernel,iterations = 2) # dilate , more the iteration more the dilation #find section
    
    _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) # get contours
    
    cnts, sorted = sort_contours(contours, "top-to-bottom")
    
    index = 0 
    dimension = 32
    for contour in sorted:

        [x,y,w,h] = contour

        #Don't plot small false positives that aren't text
        if w < 50:
            continue

        #chinese is a square, so try to get a character
        capw = w
        if(h > w):
          capw = h
        
        # draw rectangle around contour on original image
        cv2.rectangle(img,(x,y),(x+capw,y+capw),(255,0,255),2)
                
        cropped = img2[y :y + capw , x : x + capw]
        croppedBW = need_img[y :y + capw , x : x + capw]

        #cropped32 = cv2.resize(cropped, (dimension, dimension)) 
        #s = './' + folder1 + '/' + str(index) + '.png' 
        #cv2.imwrite(s , cropped32)
        
        croppedbw32 = cv2.resize(croppedBW, (dimension, dimension)) 
        s = './' + folder2 + '/'  + str(index) + '-bw.png' 
        cv2.imwrite(s , croppedbw32)
        
        index = index + 1

        
    cv2.namedWindow('captcha_result', cv2.WINDOW_NORMAL)    
    # write original image with added contours to disk  
    
    #resize for display
    height, width, channels = img.shape
    print(height)
    ratio = height / width
    newheight = 900
    newwidth = newheight / ratio
    
    imS = cv2.resize(img, (newheight, newwidth)) 
    cv2.imshow('captcha_result' , imS)
    cv2.waitKey()
    
    #cv2.imshow('captcha_result' , new_img)
    #cv2.waitKey()


file_name = sys.argv[1]
print file_name

filename_split = os.path.splitext(file_name)

folder1 = filename_split[0]
folder2 = filename_split[0] + 'bw'

createFolder(folder1)
createFolder(folder2)
        
captch_ex(file_name, folder1, folder2)