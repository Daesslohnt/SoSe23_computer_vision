class Gesture:

    def __init__(self,conditions:[lambda hand: bool]):
        self.conditions = conditions

    def is_gesture(self,hand) -> bool:
        for condition in self.conditions:
            if not condition(hand):
                return False
        return True

