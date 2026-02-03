
class FuzzController():
    def __init__(self, rules, alpha=0.01, gamma=0.99):
        self.rules = rules
        self.alpha = alpha
        self.gamma = gamma