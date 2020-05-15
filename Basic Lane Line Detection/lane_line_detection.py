import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

cap = cv2.VideoCapture('test.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    imshape = frame.shape
    vertices = np.array([[(0, imshape[0]), (0, imshape[0]/2), (imshape[1],
                                                               imshape[0]/2), (imshape[1], imshape[0])]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    rho = 2  
    theta = np.pi/180  
    threshold = 100  
    min_line_length = 40  
    max_line_gap = 5    
    line_image = np.copy(frame)*0  

    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            if(abs(math.degrees(math.atan((y2-y1)/(x2-x1)))) > 30):
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    color_edges = np.dstack((edges, edges, edges))

   
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('frame', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

