import time


class TimeController:

    def __init__(self, max_activation_s=0.1):
        self.max_activation_s = max_activation_s
        self.last_activation = time.perf_counter()

    def activate(self):
        '''
        calculates the time difference between the current time and the last_activation time.

        :return: float: time difference in seconds
        '''
        activation_time = time.perf_counter()
        delta_t = activation_time - self.last_activation
        self.last_activation = activation_time
        return delta_t

    def is_new_activation(self):
        '''
        returns True if a new activation has occurred. Otherwise, it returns False.

        :return: bool: if a new activation occurred
        '''
        return self.is_new_activation_time()[0]

    def is_new_activation_time(self):
        '''
        returns the time difference between the current activation and the previous activation in addition to a boolean indicating, if a new activation has occurred.

        :return: tuple: (bool, float): if a new activation occurred, the time difference in seconds
        '''
        delta_t = self.activate()
        if delta_t > self.max_activation_s:
            return (True, delta_t)
        return (False, delta_t)
