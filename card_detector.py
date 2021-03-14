import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import os
import argparse

path = 'resources'  #image in the resources
orb = cv2.ORB_create(nfeatures=1000)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

green = (0, 255, 0)
pink = (255, 0, 255)
thickness = 4

CARD_MAX_AREA = 3000
CARD_MIN_AREA = 230

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
args = vars(ap.parse_args())

area = []

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
cv2.namedWindow('Card Area')
cv2.resizeWindow('Parameters', 320, 120)
cv2.createTrackbar('Threshold1', 'Parameters', 30, 255, empty)
cv2.createTrackbar('Threshold2', 'Parameters', 30, 255, empty)

def find_contour(img, imgContour):
    contours, hierarchy = cv2.findContours(imgContour, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
    index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)

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

    print("Contour length", len(cnts))

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

        # (size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
        # and
        # and (hier_sort[i][3] == -1)
        # and (len(approx) == 4)
        if ((size > CARD_MIN_AREA)):
            print('[inside loop]',size)
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def find_area(img):
    maxWidth = 500
    maxHeight = 500

    ##First maze
    temp_rect = np.float32([[0, 0], [20, 20], [50, 50], [100, 100]])
    warp = warp_processing(img, temp_rect, maxWidth, maxHeight)

def preprocess_card(image, contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    x,y,w,h = cv2.boundingRect(contour)
    pts = np.float32(approx)

    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])

    warp = project_card_on_flat(image, pts, w, h)

    return (warp, (cent_x, cent_y))

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

    # warp = warp_processing(image, temp_rect, maxWidth, maxHeight)
    # # Create destination array, calculate perspective transform matrix,
    # # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warp

def warp_processing(img, temp, maxWidth, maxHeight):
    pts2 = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    matrix = cv2.getPerspectiveTransform(temp, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))
    return imgOutput


def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

### Add the key points in the image and keep in the array
desList = findDes(images)

def findID(img, desList, thres=15):
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

def detectLocation(image, contour ):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(contour)
    pts = np.float32(approx)

    average = np.sum(pts, axis=0) / len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])

    cv2.putText(image, cent_x, (0,0), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, cent_y, (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

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
        # CARD_MAX_AREA = cv2.getTrackbarPos('CARD_MAX_AREA', 'Card Area')

        # Canny edge detection
        img_canny = cv2.Canny(img_gray, threshold1, threshold2)

        # Dialation image
        kernel = np.ones((1,1))
        img_dilation = cv2.dilate(img_canny, kernel, iterations = 1)

        # cv2.imshow('Dilation', img_dilation)
        
        # Identify contour is card
        contour_sort, contour_is_card = find_cards(img_dilation)
        
        cards = []

        if len(contour_sort) != 0:
            for i in range(len(contour_sort)):
                if contour_is_card[i] == 1:
                    cards.append(preprocess_card(img_dilation, contour_sort[i]))

        for i in range(len(cards)):
            id = findID(cards[i][0], desList)

            if id != -1:
                print(classNames[id])
                cv2.putText(cards[i][0], classNames[id], (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # detectLocation(cards[i][0],contour_sort[i])

            cent_x = cards[i][1][0]
            cent_y = cards[i][1][1]

            pts = f" x:{cards[i][1][0]} y:{cards[i][1][1]}"

            cv2.putText(frame, pts, (cent_x, cent_y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('frame {}'.format(i), cards[i][0])
            cv2.imwrite("resources/{}"+str(i)+".png".format(args['image']), cards[i][0])

        # Train the first card in the found contour card
        if (cv2.waitKey(1) & 0xFF == ord('p')) and len(cards) != 0:
            print('Button P: Train the first card is pressed!')
            cv2.imwrite("resources/{}.png".format(args['image']), cards[0][0])
            cv2.imshow('frame capture', cards[0][0])

        # Show the card if it is exist in 
        if cv2.waitKey(1) & 0xFF == ord('o'):
            print('Button P: Show the trained card is pressed!')
            img = cv2.imread("resources/{}.png".format(args['image']))
            cv2.imshow('frame capture2', img)

        # find_contour(frame, img_dilation)

        # Display the resulting frame
        # resize_frame = cv2.resize(frame, (640, 480))
        # resize_img_dilation = cv2.resize(img_dilation, (1280, 720))
        # resize_img_dilation = cv2.cvtColor(resize_img_dilation, cv2.COLOR_GRAY2RGB)
        # first_frame = np.concatenate((resize_frame,resize_img_dilation), axis = 1)

        cv2.imshow('..', frame)
        cv2.imshow('.', img_dilation)

        # Exit the system
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Button Q: Thank you for using the card detector!')
            break

    # Close down the video stream
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
