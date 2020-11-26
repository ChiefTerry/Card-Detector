from model.card import Card


class Player (list):

    cardList = []

    def __init__(self, cardList = None):
        if cardList is None:
            self.cardList = []
        self.cardList = cardList

    @classmethod
    def __addCard__(cls, x, y, kind):
        card = Card(x, y , kind)
        cls.cardList.append(card)
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
