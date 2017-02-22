# Importing needed modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

class findingLaneLines(object):

    def __init__(self,challenge=False):
        #Defining challenge workaround
        self.challenge = challenge
        # Defining a kernel size and apply Gaussian smoothing
        self.kernel_size = 5
        # Defiing color mask
        self.ignore_mask_color = 255
        # Defining parameters for Canny and apply
        self.low_threshold = 50 # below color ignored
        self.high_threshold = 150 #above colors ignored
        # Defining the Hough transform parameters
        # Make a blank the same size as    our image to draw on
        self.rho = 1 # distance resolution in pixels of the Hough grid
        self.theta = np.pi/180 # angular resolution in radians of the Hough grid
        self.threshold = 25     # minimum number of votes (intersections in Hough grid cell)
        self.min_line_length = 50 #minimum number of pixels making up a line
        self.max_line_gap = 150    # maximum gap in pixels between connectable line segments
        #######################################################################
    def grayscale(self,img):
        """
        Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')
        """
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def canny(self,img, low_threshold, high_threshold):
        #Applies the Canny transform
        return cv2.Canny(img, low_threshold, high_threshold)
    def gaussian_blur(self,img, kernel_size):
        #Applies a Gaussian Noise kernel
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    def region_of_interest(self,img, vertices):
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
    
    def draw_lines(self,img, lines, color=[255, 0, 0], thickness=2):
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
                
    def split_lanes(self,lines):
        """ 
        This helper function is for seperating left and right lanes using the
        criteria that slope -/+. Left lane has negative slope and righlt lane
        has positive slopes. Also, I took a judgement call from the slope 
        distribution so i picked some ranges.
        """
        rightLane = []
        leftLane = []
        for x1,y1,x2,y2 in lines[:, 0]:
            m = float((y2) - y1) / (x2 - x1)
            if float(m >= 0.5 and m <= 0.8): 
                rightLane.append([x1,y1,x2,y2])
            elif float(m <= -0.5 and m >=-0.8):
                leftLane.append([x1,y1,x2,y2])
        
        return rightLane, leftLane
    def hough_lines(self,img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.
            
        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self.draw_lines(line_img, lines)
        return line_img
    def pipeline(self,image): 
        # Read in and grayscale the image
        gray = self.grayscale(image)
        #######################################################################
        # Gaussian smoothing 
        blur_gray = self.gaussian_blur(gray,self.kernel_size)
        #######################################################################
        # Applying Canny Edge 
        edges = self.canny(blur_gray,self.low_threshold, self.high_threshold)
        #######################################################################
        # Creating mask
        mask = np.zeros_like(edges)   
        #######################################################################
        # This time we are defining a four sided polygon to mask
        # Defining a four sided polygon to mask
        imshape = image.shape
        left_bottom = [0.098*imshape[1],imshape[0]]
        right_bottom = [imshape[1],imshape[0]]
        left_top = [0.43*imshape[1], 0.62*imshape[0]]
        right_top = [0.57*imshape[1],0.62*imshape[0]]
        #######################################################################
        # Applying the polygon mask for the region of interest 
        vertices = np.array([[(left_bottom[0],left_bottom[1]),(left_top[0], left_top[1]), (right_top[0],right_top[1]), (right_bottom[0],right_bottom[1])]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, self.ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)
        #######################################################################    
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(masked_edges, self.rho, self.theta, self.threshold, np.array([]),
                                    self.min_line_length, self.max_line_gap)
        ####################################################################### 
        """
        Challenge
        #################################################
        """
        rightLane, leftLane = self.split_lanes(lines)
        if (self.challenge):
            lines = np.array([rightLane, leftLane])
        """
        #################################################
        """
        # Drawing lines
        line_image = np.copy((image)*0)
        self.draw_lines(line_image, lines, thickness=10)
        #######################################################################         
        # Adding weight
        final = cv2.addWeighted(line_image, 0.8, image, 1, 0) 
        return final

    def process_image(self,image):
        result = self.pipeline(image)
        return result

# Instantiating classes for regular and challenge 
fl = findingLaneLines()
flChallenge = findingLaneLines(challenge=True)
####################################################################### 
# Processing solidWhiteRight.mp4 video
white_output = 'white.mp4'
clipWhite = VideoFileClip("solidWhiteRight.mp4")
white_clip = clipWhite.fl_image(fl.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False) 
#######################################################################
# Processing solidYellowLeft.mp4 video
yellow_output = 'yellow.mp4'
clipYellow = VideoFileClip("solidYellowLeft.mp4")
yellow_clip = clipYellow.fl_image(fl.process_image)
yellow_clip.write_videofile(yellow_output, audio=False) 
#######################################################################
# Processing challenge.mp4 video
challenge_output = 'CurveandShadow.mp4'
clipChallenge = VideoFileClip("challenge.mp4")
challenge_clip = clipChallenge.fl_image(flChallenge.process_image)  
challenge_clip.write_videofile(challenge_output, audio=False)   
#######################################################################    