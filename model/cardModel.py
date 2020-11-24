class Card:

    height = None
    width = None
    y = None
    x = None

    def __init__(self, x, y, width, height, isCard):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.isCard = isCard

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

    @classmethod
    def isCard(cls):
        return cls.isCard

