from model.card import Card


class Player(list):

    def __init__(self):
        self.point = 0
        self.heart = 3
        self.bang_bullet = 4
        self.click_bullet = 4
        self.num_diamond = 0
        self.num_picture = 0
        self.is_surrender = None
        self.is_dead = False

    def get_heart(self):
        return self.heart

    def get_point(self):
        return self.point

    def get_bang_bullet(self):
        return self.bang_bullet

    def get_click_bullet(self):
        return self.click_bullet

    def get_num_diamond(self):
        return self.num_diamond

    def get_num_picture(self):
        return self.num_picture

    def is_surrender(self):
        return self.is_surrender

    def is_dead(self):
        return self.is_dead

    def set_point(self, point):
        self.point = self.point + point

    def reduce_heart(self):
        if self.heart > 0:
            self.heart = self.heart - 1
        else:
            self.is_dead = True

    def increase_heart(self):
        if not self.isDead:
            self.heart = self.heart + 1

    def reduce_bang_bullet(self):
        if self.bang_bullet > 0:
            self.bang_bullet = self.bang_bullet - 1

    def reduce_click_bullet(self):
        if self.click_bullet > 0:
            self.click_bullet = self.click_bullet - 1

    def add_bang_bullet(self):
        self.bang_bullet = self.bang_bullet + 1

    def set_bullet(self):
        self.bullet = 1

    def reset_bullet(self):
        self.bullet = 0

    def increase_num_diamond(self):
        self.num_diamond = self.num_diamond + 1

    def increase_num_picture(self):
        self.num_picture = self.num_picture + 1
        if self.num_picture == 1:
            self.point = self.point + 4000
        elif self.num_picture == 2:
            self.point = self.point + 8000
        elif self.num_picture == 3:
            self.point = self.point + 18000
        elif self.num_picture == 4:
            self.point = self.point + 30000
        elif self.num_picture == 5:
            self.point = self.point + 40000
        elif self.num_picture == 6:
            self.point = self.point + 50000
        elif self.num_picture == 7:
            self.point = self.point + 50000
        elif self.num_picture == 8:
            self.point = self.point + 100000
        elif self.num_picture == 9:
            self.point = self.point + 100000
        elif self.num_picture == 10:
            self.point = self.point + 100000

    #
    # @classmethod
    # def __add__(cls):
    #     cls.cardList.append()
    # def __getCardX__(self):
    #     return self.cardList[0]
    #
    # def __getCardY__(self):
    #     return self.cardList[1]
    #
    # def __getCardKind__(self):
    #     return self.cardList[2]
    #
    # @classmethod
    # def __returnCard__(cls, index):
    #     return cls.cardList[index]
