import cv2  # Import the OpenCV library
import numpy as np  # Import Numpy library
import os
import argparse
import config


class cardDetector:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.ap = argparse.ArgumentParser()
        # self.config_card = config['CARD']
        # self.config_canny = config['CANNY_THRESHOLD']
        # self.config_window_size = config['WINDOW_SIZE']
        self.path = 'resources'
        self.config_card = {
            'thickness': 4,
            'card_max_area': 150000,
            'card_min_area': 15000,
        }
        self.config_canny = {
            'upper_threshold': 95,
            'bottom_threshold': 35
        }
        self.config_window_size = {
            'width': 320,
            'height': 120
        }

        self.orb = cv2.ORB_create(nfeatures=1000)

        self.cards = None
        self.area = None
        self.classNames = None
        self.images = None
        self.desList = None

    def empty(self):
        pass

    def __find_cards(self, thresh_image):
        """Finds all card-sized contours in a thresholded camera image.
        Returns the number of cards, and a list of card contours sorted
        from largest to smallest."""

        # Find contours and sort their indices by contour size
        cnts, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)

        # If there are no contours, do nothing
        if len(cnts) == 0:
            return [], []

        # Otherwise, initialize empty sorted contour and hierarchy lists
        cnts_sort = []
        hier_sort = []
        cnt_is_card = np.zeros(len(cnts), dtype=int)

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
            peri = cv2.arcLength(cnts_sort[i], True)
            approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * peri, True)

            if ((size < self.config_card['card_max_area']) and (size > self.config_card['card_min_area'])
                    and (hier_sort[i][3] == -1) and (len(approx) == 4)):
                cnt_is_card[i] = 1

        return cnts_sort, cnt_is_card

    def __preprocess_card(self, image, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(contour)
        pts = np.float32(approx)

        return self.__project_card_on_flat(image, pts, w, h)

    def get_center_contour(self, image, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(contour)
        pts = np.float32(approx)

        average = np.sum(pts, axis=0) / len(pts)
        cent_x = int(average[0][0])
        cent_y = int(average[0][1])

        return [cent_x, cent_y]

    def __project_card_on_flat(self, image, pts, w, h):
        temp_rect = np.zeros((4, 2), dtype="float32")

        s = np.sum(pts, axis=2)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]

        diff = np.diff(pts, axis=-1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        # Need to create an array listing points in order of
        # [top left, top right, bottom right, bottom left]
        # before doing the perspective transform

        if w <= 0.8 * h:  # If card is vertically oriented
            temp_rect[0] = tl
            temp_rect[1] = tr
            temp_rect[2] = br
            temp_rect[3] = bl

        if w >= 1.2 * h:  # If card is horizontally oriented
            temp_rect[0] = bl
            temp_rect[1] = tl
            temp_rect[2] = tr
            temp_rect[3] = br

        maxWidth = 200
        maxHeight = 500

        # warp = warp_processing(image, temp_rect, maxWidth, maxHeight)
        # # Create destination array, calculate perspective transform matrix,
        # # and warp card image
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
        M = cv2.getPerspectiveTransform(temp_rect, dst)
        warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warp

    def findID(self, img, desList, thres=15):
        kp2, des2 = self.orb.detectAndCompute(img, None)
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

        if len(matchList) != 0:
            if max(matchList) > thres:
                finalVal = matchList.index(max(matchList))

        return finalVal

    def findDes(self, images):
        self.desList = []

        for img in self.images:
            kp, des = self.orb.detectAndCompute(img, None)
            self.desList.append(des)

        return self.desList

    def get_image_from_path(self):
        myList = os.listdir(self.path)
        # print('Total Classes Detected', len(myList))
        self.images = []
        self.classNames = []

        for cl in myList:
            imgCur = cv2.imread(f'{self.path}/{cl}', 0)
            self.images.append(imgCur)
            self.classNames.append(os.path.splitext(cl)[0])

    def run(self):
        self.ap.add_argument("-i", "--image", help="path to the image file")
        args = vars(self.ap.parse_args())
        self.get_image_from_path()

        while True:
            _, frame = self.cap.read()

            # Gaussian Blur
            img_blur = cv2.GaussianBlur(frame, (5, 5), 0)
            img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

            # Adjust window and contour threshold
            cv2.namedWindow('Parameters')
            cv2.resizeWindow('Parameters', self.config_window_size['width'], self.config_window_size['height'])
            cv2.createTrackbar('Threshold1', 'Parameters', self.config_canny['bottom_threshold'], 255, self.empty)
            cv2.createTrackbar('Threshold2', 'Parameters', self.config_canny['upper_threshold'], 255, self.empty)

            # Get track position
            threshold1 = cv2.getTrackbarPos('Threshold1', 'Parameters')
            threshold2 = cv2.getTrackbarPos('Threshold2', 'Parameters')

            # Canny edge detection
            img_canny = cv2.Canny(img_gray, threshold1, threshold2)

            # Dialation image
            kernel = np.ones((1, 1))
            img_dilation = cv2.dilate(img_canny, kernel, iterations=1)

            # Identify contour is card
            contour_sort, contour_is_card = self.__find_cards(img_dilation)

            cards = []
            contours = []
            desList = self.findDes(self.images)

            if len(contour_sort) != 0:
                for i in range(len(contour_sort)):
                    if contour_is_card[i] == 1:
                        cards.append(tuple((self.__preprocess_card(img_dilation, contour_sort[i]), contour_sort[i])))
                        # cards.append(self.__preprocess_card(img_dilation, contour_sort[i]))
                        # contours.append(contour_sort[i])

            for i in range(len(cards)):
                id = self.findID(cards[i][0], desList)

                if id != -1:
                    print(self.classNames[id])
                    cv2.putText(cards[i][0], self.classNames[id], (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                pts = self.get_center_contour(img_dilation, cards[i][1])

                pts_text = f" x:{cards[i][0][1][0]} y:{cards[i][0][1][1]}"

                cv2.putText(frame, pts_text, (pts[0], pts[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('frame {}'.format(i), cards[i][0])

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

            # Display the resulting frame
            first_frame = np.concatenate((frame, img_blur), axis=1)
            second_frame = np.concatenate((img_canny, img_dilation), axis=1)
            cv2.imshow('frame', first_frame)
            cv2.imshow('frame2', second_frame)

            # Exit the system
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Button Q: Thank you for using the card detector!')
                break


if __name__ == "__main__":
    app = cardDetector()
    app.run()
