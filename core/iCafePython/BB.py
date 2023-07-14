#from iCafePython import BB
class BB:
    def __init__(self, x, y, w, h, c=None, classes=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c
        self.classes = classes
        self.label = -1
        self.score = -1

    def __repr__(self):
        return 'Bounding box at (%.1f, %.1f)' % (self.x, self.y)

    @classmethod
    def fromminmax(cls, xmin, xmax, ymin, ymax, c=None, classes=None):
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
        return cls(x, y, w, h, c, classes)

    # minmax, confidence, classtype (single int number)
    # classnumber how many classes
    @classmethod
    def fromminmaxlistclabel(cls, minmaxct, classnumber):
        xmin = minmaxct[0]
        xmax = minmaxct[2]
        ymin = minmaxct[1]
        ymax = minmaxct[3]
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
        c = minmaxct[4]
        cclasses = []
        for i in range(classnumber):
            if i == int(minmaxct[5]):
                cclasses.append(c)
            else:
                cclasses.append(0)
        return cls(x, y, w, h, c, classes=cclasses)

    def withinBB(self, x, y):
        if x >= self.x - self.w / 2 and x <= self.x + self.w / 2 and y >= self.y - self.h / 2 and y <= self.y + self.h / 2:
            return True
        else:
            return False

