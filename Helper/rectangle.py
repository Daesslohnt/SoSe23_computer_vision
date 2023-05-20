class Rectangle:
    '''
    The smallest, not rotated rectangle to contain a number of supplied points.
    '''

    def __init__(self, initial_points=[], dimensions=(1, 1)):
        self._points = initial_points
        self._dimensions = dimensions
        self._calculate_bounding_box()

    def include_point(self, x, y):
        '''
        adds a point to be included in the rectangle of points

        :param x: int: x coordinate
        :param y: int: y coordinate
        '''
        self._points.append((x, y))
        self._calculate_bounding_box()

    def _calculate_bounding_box(self):
        self.x1 = int(sorted(self._points, key=lambda p: p[0])[0][0] * self._dimensions[0])
        self.y1 = int(sorted(self._points, key=lambda p: p[1])[0][1] * self._dimensions[1])
        self.x2 = int(sorted(self._points, key=lambda p: p[0], reverse=True)[0][0] * self._dimensions[0])
        self.y2 = int(sorted(self._points, key=lambda p: p[1], reverse=True)[0][1] * self._dimensions[1])

    def get_bounding_box(self):
        '''
        returns the upper left and lower right coordinates of the rectangles bounding box

        :return: list:tuple:int: The bounding box of the rectangle
        '''
        return [(self.x1, self.y1), (self.x2, self.y2)]

    def collides(self, other):
        '''
        checks for collision with the supplied rectangle

        :param other: Rectangle
        :return: bool: If the rectangle collides with the supplied rectangle
        '''
        self_bound = self.get_bounding_box()
        other_bound = other.get_bounding_box()

        return \
                self_bound[0][0] < other_bound[1][0] and \
                self_bound[0][1] < other_bound[1][1] and \
                self_bound[1][0] > other_bound[0][0] and \
                self_bound[1][1] > other_bound[0][1]

    def collides_y(self, other):
        '''
        checks for collisions on the y-axis

        :param other: Rectangle
        :return: bool: If the rectangles overlap on the y axis
        '''
        self_bound = self.get_bounding_box()
        other_bound = other.get_bounding_box()

        return \
                self_bound[0][1] < other_bound[1][1] and \
                self_bound[1][1] > other_bound[0][1]

    def collides_x(self, other):
        '''
        checks for collisions on the x-axis

        :param other: Rectangle
        :return: bool: If the rectangles overlap on the x axis
        '''
        self_bound = self.get_bounding_box()
        other_bound = other.get_bounding_box()

        return \
                self_bound[0][0] < other_bound[1][0] and \
                self_bound[1][0] > other_bound[0][0]
