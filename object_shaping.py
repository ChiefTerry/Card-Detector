import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library

cap = cv2.VideoCapture(0)
green = (0, 255, 0)
pink = (255, 0, 255)
thickness = 4

def empty():
    pass

cv2.namedWindow('Parameters')
cv2.resizeWindow('Parameters', 640, 240)
cv2.createTrackbar('Threshold1', 'Parameters', 0, 255, empty)
cv2.createTrackbar('Threshold2', 'Parameters', 255, 255, empty)

def point_prediction(img, approx, area, point, width, height):
    x = point[0]
    y = point[1]
    
    # Prediction
    if len(approx) == 8:
        cv2.putText(img, "Prediction: Cylinder", (x + width + 20, y + 70), \
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, green, 2)
    elif len(approx) == 6:
        cv2.putText(img, "Prediction: Cuboid", (x + width + 20, y + 70), \
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, green, 2)
    elif len(approx) > 8:
        cv2.putText(img, "Prediction: Unknown", (x + width + 20, y + 70), \
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, green, 2)                

def find_contour(img, imgContour):
    contours, hierarchy = cv2.findContours(imgContour, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
    # cv2.drawContours(img, contours, -1, (0,255,0), 7)

    if len(contours) > 1:
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 9000:
                # Draw contour
                # cv2.drawContours(img, contour, -1, green, thickness)

                # Get perimeter and approximation of boundary
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                # print(len(approx))
                
                # Apply bounding rectangle
                x, y, width, height = cv2.boundingRect(approx)
                initial_point = (x, y)
                rec_width = x + width
                rec_height = y + height        
                cv2.rectangle(img, initial_point, (rec_width, rec_height), pink, 7)

                # Display text
                cv2.putText(img, "Points: " + str(len(approx)), (x + width + 20, y + 20), \
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, green, 2)
                cv2.putText(img, "Area: " + str(int(area)), (x + width + 20, y + 45), \
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, green, 2)

                point_prediction(img, approx, area, initial_point, width, height)
                

while True:
    ret, frame = cap.read()
    # imgContour = frame.copy()
    
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    # Gaussian Blur
    img_blur = cv2.GaussianBlur(frame, (7, 7), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    # Get track position
    threshold1 = cv2.getTrackbarPos('Threshold1', 'Parameters')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'Parameters')

    # Canny edge detection
    img_canny = cv2.Canny(img_gray, threshold1, threshold2)

    # Dialation image
    kernel = np.ones((5,5))
    img_dilation = cv2.dilate(img_canny, kernel, iterations = 1)

    find_contour(frame, img_dilation)
    # contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
    # cv2.drawContours(frame, contours, -1, (0,255,0), 7)

    # Display the resulting frame
    # img_combine = np.concatenate([frame, img_dilation], axis = 1)
    cv2.imshow('frame',frame)
    cv2.imshow('frame2', img_dilation)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Close down the video stream
cap.release()
cv2.destroyAllWindows()