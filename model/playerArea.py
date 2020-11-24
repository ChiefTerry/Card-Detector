class Area :
    cardList = None

    def __init__(self, x, y, width, height):
        self.cardList = []
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    # getter method
    @classmethod
    def get_x(cls):
        return cls.x

    @classmethod
    def get_y(cls):
        return cls.y

    @classmethod
    def get_width(cls):
        return cls.width

    @classmethod
    def get_height(cls):
        return cls.height

    def clearList(self):
        self.cardList.clear()

    def addCard(self, playerArea):
        self.cardList.append(playerArea)