import cv2

# Reading the video we want to analyze
capture = cv2.VideoCapture('people_video.avi') #Uncomment this If you want to detect people movement
# capture = cv2.VideoCapture('cars_video.avi') #Uncomment this if you want to detect cars movement

# This is for saving video in a .AVI format
video_fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

# This will save result video to folder. (video properties: 30 fps and 1280x720 resolution)
output = cv2.VideoWriter("output.avi", video_fourcc, 30.0, (1280, 720))

# Here we take 2 frames in order to compare them and find out, what is changed/moved
ret, frame1 = capture.read()
ret, frame2 = capture.read()

# While video is playing
while capture.isOpened():
    # Calculates absolute difference between two frames
    difference = cv2.absdiff(frame1, frame2)
    # Converting frame difference to grayscale format
    gray_scale = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    # Applying gaussian blur to each frame with kernel value 5
    kernel = 5
    blurry = cv2.GaussianBlur(gray_scale, (kernel, kernel), 0)
    # This function changes color of pixel to 20 if it's less than 127, else it changes it to 255
    _, threshold = cv2.threshold(blurry, 20, 255, cv2.THRESH_BINARY)
    # dilate function used to accentuate features
    dilate = cv2.dilate(threshold, None, iterations=2)
    # this function finds contours in a binary image
    contours, _ = cv2.findContours(
        dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Here we will get cordinates of each contour
        # x-cordinate, y-cordinate, w-width, h-height
        (x, y, w, h) = cv2.boundingRect(contour)

        #  Takes contour areas that are not less than 900
        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 0, 0), 2) # Here we define properties of rectangle

    # Here we rezise each frame resolution
    image = cv2.resize(frame1, (1280, 720))
    output.write(image)
    #This will be the output window after running the code
    cv2.imshow("Output", frame1) # After applying the contour the result will be saved to frame1
    frame1 = frame2
    ret, frame2 = capture.read() # We are reading new frame and storing it frame2 and before getting new frame, we are assigning its value to frame1
    # If you press 'q' you will exit the output video
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
capture.release()
output.release()