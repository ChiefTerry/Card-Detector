class Card(object):
    y = None
    x = None
    type = None

    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type

    # getter method
    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_type(self):
        return self.type

    @classmethod
    def __set_type__(self, type):
        self.type = type