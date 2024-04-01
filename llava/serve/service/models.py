from collections import OrderedDict


class LRUCache(OrderedDict):

    def __init__(self, maxsize):
        self.maxsize = maxsize
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        elif len(self) >= self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

        super().__setitem__(key, value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


class Conversation:
    def __init__(self):
        self.messages = []

    def __str__(self):
        return str(self.messages)

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_messages(self):
        return self.messages
