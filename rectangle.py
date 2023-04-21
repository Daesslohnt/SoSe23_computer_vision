class Rectangle:

    def __init__(self, initial_points=[], dimensions=(1, 1)):
        self.points = initial_points
        self.dimensions = dimensions
        self._calculate_bounding_box()

    def include_point(self, x, y):
        self.points.append((x, y))
        self._calculate_bounding_box()

    def _calculate_bounding_box(self):
        self.x1 = int(sorted(self.points, key=lambda p: p[0])[0][0] * self.dimensions[0])
        self.y1 = int(sorted(self.points, key=lambda p: p[1])[0][1] * self.dimensions[1])
        self.x2 = int(sorted(self.points, key=lambda p: p[0], reverse=True)[0][0] * self.dimensions[0])
        self.y2 = int(sorted(self.points, key=lambda p: p[1], reverse=True)[0][1] * self.dimensions[1])

    def get_bounding_box(self):
        return [(self.x1, self.y1), (self.x2, self.y2)]

    def collides(self, other):
        self_bound = self.get_bounding_box()
        other_bound = other.get_bounding_box()

        return \
                self_bound[0][0] < other_bound[1][0] and \
                self_bound[0][1] < other_bound[1][1] and \
                self_bound[1][0] > other_bound[0][0] and \
                self_bound[1][1] > other_bound[0][1]

    def collides_y(self, other):
        self_bound = self.get_bounding_box()
        other_bound = other.get_bounding_box()

        return \
                self_bound[0][1] < other_bound[1][1] and \
                self_bound[1][1] > other_bound[0][1]
