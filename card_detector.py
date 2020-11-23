import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import os

path = 'resources'  #image in the resources
orb = cv2.ORB_create(nfeatures=1000)

cap = cv2.VideoCapture(0)

green = (0, 255, 0)
pink = (255, 0, 255)
thickness = 4

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

#### Import Images in the resources 
images = []
classNames = []
myList = os.listdir(path)
print('Total Classes Detected', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])

def empty():
    pass

cv2.namedWindow('Parameters')
cv2.resizeWindow('Parameters', 640, 240)
cv2.createTrackbar('Threshold1', 'Parameters', 35, 255, empty)
cv2.createTrackbar('Threshold2', 'Parameters', 95, 255, empty)

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
    index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
    # cv2.drawContours(img, contours, -1, (0,255,0), 7)

    if len(contours) > 1:
        contours_sort = []
        hierarchy_sort = []
        card_in_contour = np.zeros(len(contours), 'uint8')

        for i in index_sort:
            contours_sort.append(contours[i])
            hierarchy_sort.append(hierarchy[0][i])

        for contour in contours:
            area = cv2.contourArea(contour)
            
            if CARD_MIN_AREA < area < CARD_MAX_AREA:
                # Get perimeter and approximation of boundary
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                print(len(approx))
                
                # Apply bounding rectangle
                x, y, width, height = cv2.boundingRect(approx)
                initial_point = (x, y)
                rec_width = x + width
                rec_height = y + height        

                # Display text
                if len(approx) == 4:
                    cv2.rectangle(img, initial_point, (rec_width, rec_height), pink, 7)
                    cv2.putText(img, "Points: " + str(len(approx)), (x + width + 20, y + 20), \
                        cv2.FONT_HERSHEY_COMPLEX, 1.0, green, 2)
                    cv2.putText(img, "Area: " + str(int(area)), (x + width + 20, y + 45), \
                        cv2.FONT_HERSHEY_COMPLEX, 1.0, green, 2)
                    
                
def find_cards(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""

    # Find contours and sort their indices by contour size
    cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []
    
    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def preprocess_card(image, contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    x,y,w,h = cv2.boundingRect(contour)
    pts = np.float32(approx)

    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])

    warp = project_card_on_flat(image, pts, w, h)

    return warp

def project_card_on_flat(image, pts, w, h):
    temp_rect = np.zeros((4,2), dtype = "float32")

    s = np.sum(pts, axis = 2)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    maxWidth = 200
    maxHeight = 500
    
    # Create destination array, calculate perspective transform matrix, 
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    # cv2.imshow('frame3', warp)

    return warp

def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

### Add the key points in the image and keep in the array
desList = findDes(images)

def findID(img, desList, thres=90):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    # print(matchList)
    if len(matchList) != 0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal


def main():
    while True:
        ret, frame = cap.read()
        # imgContour = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Gaussian Blur
        img_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

        # Get track position
        threshold1 = cv2.getTrackbarPos('Threshold1', 'Parameters')
        threshold2 = cv2.getTrackbarPos('Threshold2', 'Parameters')

        # Canny edge detection
        img_canny = cv2.Canny(img_gray, threshold1, threshold2)

        # Dialation image
        kernel = np.ones((3,3), np.float32)/9
        img_dilation = cv2.dilate(img_canny, kernel, iterations = 1)

        contour_sort, contour_is_card = find_cards(img_dilation)

        cards = []

        if len(contour_sort) != 0:
            
            for i in range(len(contour_sort)):
                if contour_is_card[i] == 1:
                    # cv2.imshow(str(i), preprocess_card(img_gray, contour_sort[i]))
                    cards.append(preprocess_card(gray, contour_sort[i]))
                    # cv2.imshow('frame3', warp)
                    # pass

        # Process and classify card        
        # for card in cards:
            
        #         cv2.putText(card, classNames[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        #         print(classNames[id])

        # preprocess_card(img_dilation, contour_sort[0])



        # Display the resulting frame
        cv2.imshow('frame',gray)
        # cv2.imshow('frame2',img_dilation)
        for i in range(len(cards)):
            id = findID(cards[i], desList)
            if id != -1:
                cv2.putText(cards[i], classNames[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                cv2.imshow('frame {}'.format(i), cards[i])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close down the video stream
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
