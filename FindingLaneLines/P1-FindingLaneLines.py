# Importing python modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

rightAvrg = []
leftAvrg = []
#os.chdir("test_images/")
images = ['solidWhiteCurve.jpg',
          'solidWhiteRight.jpg',
          'solidYellowCurve.jpg',
          'solidYellowCurve2.jpg',
          'solidYellowLeft.jpg',
          'whiteCarLaneSwitch.jpg']

directory = r"C:\Server\Classes\SelfDrivingCar\CarND-LaneLines-P1\test_images"


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """

    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def split_lanes(lines):
    """ Takes an array of hough lines and separates them by +/- slope.
        The y-axis is inverted in matplotlib, so the calculated positive slopes will be right
        lane lines and negative slopes will be left lanes. """
    right = []
    left = []
    for x1,y1,x2,y2 in lines[:, 0]:
        m = float((y2) - y1) / (x2 - x1)
        if float(m >= 0.5 and m <= 0.8): 
            right.append([x1,y1,x2,y2])
        elif float(m <= -0.5 and m >=-0.8):
            left.append([x1,y1,x2,y2])
    
    return right, left

def average(lanes):
    count = len(lanes)
    total = 0
    for lane in lanes:
        total = total + lane[4]
    if (count == 0):
        return 0
    else:
        avrg = float(total)/count
    return avrg
def averageFinal(lanes):
    count = len(lanes)
    total = 0
    for lane in lanes:
        plt.scatter(np.linspace(-1, 1,1),lane)
        total = total + lane
    avrg = float(total)/count
    return avrg
def pipeline(image): 
    # Read in and grayscale the image
    gray = grayscale(image)
    #plt.imshow(gray, cmap='gray')
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    #blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    blur_gray = gaussian_blur(gray,kernel_size)
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    #edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    edges = canny(blur_gray,low_threshold, high_threshold)
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   
    
    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    
    left_bottom = [0.098*imshape[1],imshape[0]]
    right_bottom = [imshape[1],imshape[0]]
    left_top = [0.43*imshape[1], 0.62*imshape[0]]
    right_top = [0.57*imshape[1],0.62*imshape[0]]
    #vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
    vertices = np.array([[(left_bottom[0],left_bottom[1]),(left_top[0], left_top[1]), (right_top[0],right_top[1]), (right_bottom[0],right_bottom[1])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    #masked_edges = region_of_interest(image,vertices)
    
    # Define the Hough transform parameters
    # Make a blank the same size as    our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 25     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50 #minimum number of pixels making up a line
    max_line_gap = 120    # maximum gap in pixels between connectable line segments
    #line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    #line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    """
    Improvement
    #################################################
    """

    #print lines
    right, left = split_lanes(lines)
    #print "Size: ",len(right),"-->",len(left)
    #print "M:", right[0][4], " -->", left[0][4]
    #print "Right Average: ", average(right)
    #print "Left Average: ", average(left)
    #rightAvrg.append(average(right))
    #leftAvrg.append(average(left))
    #lines = np.concatenate((right, left))
    lines = np.array([right, left])
    """
    #################################################
    """
    
    
    
    line_image = np.copy((image)*0)
    
    draw_lines(line_image, lines, thickness=10)
    line_image = region_of_interest(line_image, vertices)
    final = cv2.addWeighted(line_image, 0.8, image, 1, 0) 

    # Iterate over the output "lines" and draw lines on a blank image
    #draw_lines(line_image, lines)
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    
    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(line_image, 0.8, image, 1, 0) 

    return final
    """
    x = [left_bottom[0], right_bottom[0], right_top[0], left_top[0],left_bottom[0]]
    y = [left_bottom[1], right_bottom[1], right_top[1], left_top[1],left_bottom[1]]


    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Edge Detection', fontsize=20)
    fig.set_size_inches(15, 6)
    plt.plot(x, y, 'b--', lw=4)
    ax1.imshow(lines_edges)
    ax2.imshow(image)    
    """

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    #clip1.write_videofile(white_output, audio=False)
    result = pipeline(image)
    """Flips an image vertically """
    #return image[::-1] # remember that image is a numpy array
    return result

#white_output = 'white.mp4'
#white_output = 'yellow.mp4'
white_output = 'CurveandShadow.mp4'
#clip1 = VideoFileClip("..\CarND-LaneLines-P1\solidWhiteRight.mp4")
#clip1 = VideoFileClip("..\CarND-LaneLines-P1\solidYellowLeft.mp4")
clip1 = VideoFileClip("..\CarND-LaneLines-P1\challenge.mp4")

white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time white_clip.write_videofile(white_output, audio=False)
white_clip.write_videofile(white_output, audio=False)
#print "Rigt Average: ", averageFinal(rightAvrg)
#print "Left Average: ", averageFinal(leftAvrg)

for imageList in images:
    #process_image("test")
    # Read in the image and print out some stats
    #image =  mpimg.imread('image.jpg')
    print "Image: ", imageList
    image = mpimg.imread(directory+"\\"+imageList)
    print('This image is: ',type(image),
          'width dimensions: ', image.shape)


    # Read in and grayscale the image
    gray = grayscale(image)
    #plt.imshow(gray, cmap='gray')
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    #blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    blur_gray = gaussian_blur(gray,kernel_size)
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    #edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    edges = canny(blur_gray,low_threshold, high_threshold)
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   
    
    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    
    left_bottom = [0.098*imshape[1],imshape[0]]
    right_bottom = [imshape[1],imshape[0]]
    left_top = [0.43*imshape[1], 0.62*imshape[0]]
    right_top = [0.57*imshape[1],0.62*imshape[0]]
    #vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
    vertices = np.array([[(left_bottom[0],left_bottom[1]),(left_top[0], left_top[1]), (right_top[0],right_top[1]), (right_bottom[0],right_bottom[1])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    #masked_edges = region_of_interest(image,vertices)
    
    # Define the Hough transform parameters
    # Make a blank the same size as    our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 5     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 15    # maximum gap in pixels between connectable line segments
    #line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(line_image, lines)
    #lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    # Iterate over the output "lines" and draw lines on a blank image

    #draw_lines(line_image, lines)
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    
    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 

    line_edges = region_of_interest(lines_edges, vertices)
    
    x = [left_bottom[0], right_bottom[0], right_top[0], left_top[0],left_bottom[0]]
    y = [left_bottom[1], right_bottom[1], right_top[1], left_top[1],left_bottom[1]]


    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Edge Detection', fontsize=20)
    fig.set_size_inches(15, 6)
    plt.plot(x, y, 'b--', lw=4)
    ax1.imshow(lines_edges)
    ax2.imshow(image)
