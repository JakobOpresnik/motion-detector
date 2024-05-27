import cv2

# initialize camera (0 - default camera)
camera = cv2.VideoCapture(0)

# read first 2 frames --> used to detect changes (motion) between them
ret, frame1 = camera.read()
ret, frame2 = camera.read()

# loop runs as long as the camera is open
while camera.isOpened():
    # absolute difference between 1st and current frame
    # highlights areas that have changed between the 2 frames --> motion
    diff = cv2.absdiff(frame1, frame2)

    # conversion to grayscale
    # smaller size while still sufficient for motion detection
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur
    # reduces noise & smoothes out the image
    # easier to detect significant changes
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # conversion to binary image
    # pixels with values above 20 are set to white - 255, and those below to black - 0
    # isolates regions with significant change by clearly separating motion areas from the background
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # dilation fills small gaps & connects adjacent regions of motion
    dilated = cv2.dilate(thresh, kernel=None, iterations=3)

    # find contours (boundaries) of white regions
    # contours represent areas where motion is detected
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw bounding rectangles around found contours
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 700:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # display result
    cv2.imshow('Motion Detector', frame1)

    # update frames
    frame1 = frame2
    # read new frame from camera
    ret, frame2 = camera.read()

    # press ESC to close camera
    if cv2.waitKey(10) == 27:
        break

# close all windows
camera.release()
cv2.destroyAllWindows()