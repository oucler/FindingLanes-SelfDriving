# Finding Lane Lines

The goal of this project is to detect lines of any color on the road for provided videos – one of the videos with solid white line/dashed white lines and the second video is with solid yellow line and dashed white lines. Also, there is a challenge video that stresses different perspective such as light contrasts and the road being curvier compared to the previous videos. Basically, the project is to draw lines between lanes using computer vision techniques.   

- Solid White Lines Video Result:

  ![White Solid Lines](videos/white.gif)

- Solid Yellow Lines Video Result:

  ![White Solid Lines](videos/yellow.gif)
 
- [Project Video](https://www.youtube.com/watch?v=Qn0w2xHP8U0)

## Table of Contents ##
- [Reflection](#reflection)
- [Shortcomings](#shortcomings)
- [Improvements](#improvements)



## Reflection <a name="reflection"></a>
Initial step is to convert into grayscale (one color channel 8bits) which is helpful for detecting edge in the image then applying Gaussian smoothing for removing the noise lastly Canny Edge detection finds out intensity gradient which leads to determining strong and weak edges. Weak edges not connected to strong edges to be deleted. Then, selecting only the interest of the region following with conversion of Hough Transform for further filtering the features causing noise. Finally, weighted sum between original and modified images are calculated.  
Pipeline is implemented in 6 steps:

1.	Converting the image into gray scale
2.	Applying Gaussian smoothing
3.	Canny Edge detection 
4.	Defining the region of the interest parameters
5.	Applying parameters of the interest
6.	Converting image into Hough Transform
7.	Adding weights  


## Shortcomings <a name="shortcomings"></a>
The implementation is completed under some assumptions and if those assumptions are not met accuracy of the lane line detection will drop. 
Assumptions:
a)	No light contrast such as sunny and shadow regions of the video
b)	Lane lines are closer to straight lines
c)	Day light and sunny weather – sunset and sunrise will impact the result
d)	Well defined lane lines and there are always two lane lines separating the road
There are many other scenarios that not covered in this implementations for instance if the video was taken during night time the algorithm requires changes or if it was a rainy day that would make it challenging to detect lane lines.  


## Improvements <a name="improvements"></a>
Improvements would be implementing an algorithm to meet expectations of the assumptions. However, that would be the ideal case that is not the scope of this project. At the challenge video, there were drawn lines with different slopes so I manually measure the slope between right and left lanes to see what the expected slopes then filtered out the outliers. For better accuracy that will require automated way of filtering correct slope values. 
