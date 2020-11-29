import json

import cv2  # Import the OpenCV library
import numpy as np  # Import Numpy library
import os
import argparse

from model import player
from model.card import Card
from model.player import Player
from model.typeCard import Type

green = (0, 255, 0)
pink = (255, 0, 255)


class heart:

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

        self.config_damaged = {
            'max_thickness': 14,
            'min_thickness': 12,
            'damaged_max_area': 25000,
            'damaged_min_area': 10000,
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

    def __find_cards(self, thresh_image, img):
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
            if (len(approx) >= 12 and len(approx) <= 14) and (10000 < size) and (hier_sort[i][3] == -1):
                # Apply bounding rectangle
                x, y, width, height = cv2.boundingRect(approx)
                initial_point = (x, y)
                rec_width = x + width
                rec_height = y + height
                cv2.rectangle(img, initial_point, (rec_width, rec_height), pink, 7)

                # Display text
                cv2.putText(img, "Points: " + str(len(approx)), (x + width + 20, y + 20), \
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, green, 2)
                cv2.putText(img, "Area: " + str(int(size)), (x + width + 20, y + 45), \
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, green, 2)
                cnt_is_card[i] = 1

        return cnts_sort, cnt_is_card

    def __preprocess_card(self, image, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(contour)
        pts = np.float32(approx)

        pts1 = np.float32([[x, y], [x + w, y], [x , y + h], [x + w, y + h]])
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(image, matrix, (w, h))

        for x in range(0, 4):
            cv2.circle(image, (pts1[x][0], pts1[x][1]), 15, (0, 255, 0), cv2.FILLED)

        return imgOutput

    def get_center_contour(self, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(contour)
        pts = np.float32(approx)

        average = np.sum(pts, axis=0) / len(pts)
        cent_x = int(average[0][0])
        cent_y = int(average[0][1])

        return cent_x, cent_y

    def draw_rectangle_test(self, image, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

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
            temp_rect[0] = tl - 5
            temp_rect[1] = tr - 5
            temp_rect[2] = br + 5
            temp_rect[3] = bl + 5

        if w >= 1.2 * h:  # If card is horizontally oriented
            temp_rect[0] = bl - 5
            temp_rect[1] = tl- 5
            temp_rect[2] = tr + 5
            temp_rect[3] = br + 5

        maxWidth = 200
        maxHeight = 500

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

    def process_points(self):
        for i in range(len(self.players)):
            player = self.players[i]
            for f in range(len(self.players[i])):
                nameCard = self.players[i][f].get_type()
                if nameCard in self.point_dictionary:
                    player.set_point(self.point_dictionary[nameCard])
                    # increase number of diamond
                    if nameCard == "diamond_1" or nameCard == "diamond_5" or nameCard == "diamond_10":
                        player.increase_num_diamond()

                elif nameCard == "add_bullet":
                    player.add_bang_bullet()

                elif nameCard == "bang_dilation":
                    player.reduce_bang_bullet()

                elif nameCard == "click_dilation":
                    player.reduce_click_bullet()

                elif nameCard == "cure":
                    player.increase_heart()

                elif nameCard == "diamond_1":
                    player.increase_heart()

                elif nameCard == "picture" and nameCard == "picture1" and nameCard == "picture2" and \
                        nameCard == "picture3" and nameCard == "picture4" and nameCard == "pictur5" and \
                        nameCard == "picture6" and nameCard == "picture7" and nameCard == "picture8" and \
                        nameCard == "picture9":
                    player.increase_num_picture()

    def importJson(self):
        data = {}
        for i in range(len(self.players)):
            player_num = 'player' + str(i)
            player = self.players[i]
            data[player_num] = {}
            data[player_num]["cards"] = {}
            for f in range(len(self.players[i])):
                card = self.players[i][f]
                card_key = 'card' + str(f)
                data[player_num]["cards"][card_key] = {}

                # add attribute of the cards
                data[player_num]["cards"][card_key]['x'] = card.get_x()
                data[player_num]["cards"][card_key]['y'] = card.get_y()
                data[player_num]["cards"][card_key]['type'] = card.get_type()

            data[player_num]['heart'] = player.get_heart()
            data[player_num]['point'] = player.get_point()
            data[player_num]['bang_bullet'] = player.get_bang_bullet()
            data[player_num]['click_bullet'] = player.get_click_bullet()
            data[player_num]['num_diamond'] = player.get_num_diamond()
            data[player_num]['num_picture'] = player.get_num_picture()
            # data[player_num]['is_surrender'] = player.is_surrender()
            # data[player_num]['is_dead'] = player.is_dead()

            # append card in list cards
        print(data)

        with open('data.txt', 'w') as outfile:
            json.dump(data, outfile)

    def run(self):
        self.ap.add_argument("-i", "--image", help="path to the image file")
        args = vars(self.ap.parse_args())
        self.get_image_from_path()

        # Create the pleayer area
        self.create_players()

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
            contour_sort, contour_is_card = self.__find_cards(img_dilation, frame)

            # Store the images in the resources with the key points
            desList = self.findDes(self.images)

            # renew card list in the player list
            self.renew_card_list()

            if len(contour_sort) != 0:
                for i in range(len(contour_sort)):
                    if contour_is_card[i] == 1:
                        card = self.__preprocess_card(img_dilation, contour_sort[i])

                        id = self.findID(card, desList)

                        if id != -1:
                            print(self.classNames[id])
                            cv2.putText(card, self.classNames[id], (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 255, 0), 2)

                        pts = self.get_center_contour(contour_sort[i])
                        self.draw_rectangle_test(frame, contour_sort[i])

                        # Put card in the player card list
                        player = self.get_player_id(pts, self.classNames[id])
                        cv2.putText(frame, 'player {}'.format(player), (pts[0], pts[1]), cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 255, 0), 2)
                        cv2.imshow('frame {}'.format(i), card)

            # testing add bullet
            # before adding
            print("Before adding: ")
            for i in range(len(self.players)):
                player = self.players[i]
                print(player.point)

            self.process_points()

            # after adding
            print("After adding: ")
            for i in range(len(self.players)):
                player = self.players[i]
                print("Player " + str(i) + "has : " + str(player.bang_bullet))

            #
            # # Train the first card in the found contour card
            # if (cv2.waitKey(1) & 0xFF == ord('p')) and len(self.cards) != 0:
            #     print('Button P: Train the first card is pressed!')
            #     cv2.imwrite("resources/{}.png".format(args['image']), self.cards[0][0])
            #     cv2.imshow('frame capture', self.cards[0][0])

            self.importJson()

            # Show the card if it is exist in
            if cv2.waitKey(1) & 0xFF == ord('o'):
                print('Button P: Show the trained card is pressed!')
                img = cv2.imread("resources/{}.png".format(args['image']))
                cv2.imshow('frame capture2', img)

            # Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.imshow('frame2', img_dilation)

            # Exit the system
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Button Q: Thank you for using the card detector!')
                break


if __name__ == "__main__":
    app = heart()
    app.run()
