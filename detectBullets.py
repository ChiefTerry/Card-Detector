import json

import cv2  # Import the OpenCV library
import numpy as np  # Import Numpy library
import os
import argparse

from model.card import Card
from model.player import Player


class cardDetector:

    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        # self.cap = argparse.ArgumentParser()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.path = 'resources'
        self.config_card = {
            # Detect small card with high distance
            'thickness': 4,
            'card_max_area': 3000,  # Origin is 200000
            'card_min_area': 150,  # Origin is 15000
        }
        self.config_canny = {
            'upper_threshold': 30,
            'bottom_threshold': 30
        }
        self.config_window_size = {
            'width': 640,
            'height': 240
        }

        self.point_dictionary = {
            '5_dollar': 5000,
            '10_dollar': 10000,
            '20_dollar': 20000,
            'diamond_1': 1000,
            'diamond_5': 5000,
            'diamond_10': 10000,
        }

        self.orb = cv2.ORB_create(nfeatures=1000)

        self.players = None
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

            print(size)
            # (size < self.config_card['card_max_area']) and
            # and (hier_sort[i][3] == -1) and (len(approx) == 4)
            if size > self.config_card['card_min_area'] and (hier_sort[i][3] == -1):
                cnt_is_card[i] = 1

        # remember to delete card area
        return cnts_sort, cnt_is_card

    def __preprocess_card(self, image, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(contour)
        pts = np.float32(approx)

        return self.__project_card_on_flat(image, pts, w, h)

    def draw_rectangle_test(self, image, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    def __project_card_on_flat(self, image, pts, w, h):
        temp_rect = np.zeros((4, 2), dtype="float32")
        points = np.zeros((4, 2), dtype="float32")

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

        points[0] = tl
        points[1] = tr
        points[2] = br
        points[3] = bl

        # Pop up frame
        maxWidth = 200
        maxHeight = 500

        # # Create destination array, calculate perspective transform matrix,
        # # and warp card image
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
        M = cv2.getPerspectiveTransform(temp_rect, dst)
        warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warp, points

    def findID(self, img, desList, thres=13):
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

        print(str(finalVal) + " final val")
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

    def get_player_id(self, pts=(0, 0), kind=""):
        x = pts[0]
        y = pts[1]
        card = Card(x, y, kind)

        if 0 < x <= 600 and 0 < y <= 400:
            self.players[0].append(card)
            return 0
        elif 600 < x <= 1200 and 0 < y <= 400:
            self.players[1].append(card)
            return 1
        elif 0 < x <= 600 and 400 < y <= 800:
            self.players[0].append(card)
            return 2
        elif 600 < x <= 1200 and 400 < y <= 800:
            self.players[0].append(card)
            return 3
        return 0

    def create_players(self):
        self.players = []
        for i in range(4):
            self.players.append(Player())

    def renew_card_list(self):
        for i in range(4):
            self.players[i].clear()

    def run(self):

        self.get_image_from_path()

        # Create the pleayer area
        self.create_players()

        while True:
            ret_, frame = self.cap.read()

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
            # remmber to delete size in card
            contour_sort, contour_is_card = self.__find_cards(img_dilation)

            # Store the images in the resources with the key points
            desList = self.findDes(self.images)

            # renew card list in the player list
            self.renew_card_list()
            # New list of data insert
            data = {}
            data['bullets'] = []

            if len(contour_sort) != 0:
                for i in range(len(contour_sort)):
                    if contour_is_card[i] == 1:
                        card, points = self.__preprocess_card(img_dilation, contour_sort[i])

                        id = self.findID(card, desList)

                        if id != -1:
                            print(self.classNames[id])
                            cv2.putText(card, self.classNames[id], (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 255, 0), 2)

                            self.draw_rectangle_test(frame, contour_sort[i])

                            bullets = {}
                            bullets['1'] = [points[0][0], points[0][1]]
                            bullets['2'] = [points[1][0], points[1][1]]
                            bullets['3'] = [points[2][0], points[2][1]]
                            bullets['4'] = [points[3][0], points[3][1]]
                            bullets['type'] = self.classNames[id]

                            data['bullets'].append(bullets)
                            # Put card in the player card list
                            cv2.putText(frame, "1", (points[0][0], points[0][1]), cv2.FONT_HERSHEY_COMPLEX,
                                        1,(0, 255, 0), 2)
                            cv2.putText(frame, "2", (points[1][0], points[1][1]), cv2.FONT_HERSHEY_COMPLEX,
                                        1, (0, 255, 0), 2)
                            cv2.putText(frame, "3", (points[2][0], points[2][1]), cv2.FONT_HERSHEY_COMPLEX,
                                        1, (0, 255, 0), 2)
                            cv2.putText(frame, "4", (points[3][0], points[3][1]), cv2.FONT_HERSHEY_COMPLEX,
                                        1, (0, 255, 0), 2)
                            # cv2.imshow('frame {}'.format(i), card)

                # write file bullet json in the unity project
                with open('/Users/tranmachsohan/Desktop/boardgame-ar-project/Assets/StreamingAssets/bullets.json',
                          'w') as outfile:
                    json.dump(data, outfile)

            # Show the card if it is exist in
            # if cv2.waitKey(1) & 0xFF == ord('o'):
            # print('Button P: Show the trained card is pressed!')
            # img = cv2.imread("resources/{}.png".format(args['image']))
            # cv2.imshow('frame capture2', img)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.imshow('frame2', img_dilation)

            # Exit the system
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Button Q: Thank you for using the card detector!')
                break


if __name__ == "__main__":
    app = cardDetector()
    app.run()
