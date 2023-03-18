class BaseReplay():
    def __init__(self, max_size, batch_size):
        self.batch_size = batch_size
        self.max_size = int(max_size)
        self.size = 0

    def add(self, data, priority):
        pass

    def sample(self):
        pass

    def priority_update(self, *args):
        pass