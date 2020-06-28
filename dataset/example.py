import collections


class Example(collections.UserDict):
    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]

    def __contains__(self, key):
        return str(key) in self.data

    def __setitem__(self, key, item):
        self.data[str(key)] = item

    def __getattr__(self, key):
        return self.data[str(key)]

    def __getstate__(self):
        return self.data

    def __setstate__(self, state):
        self.data = state




if __name__ == '__main__':
    example = Example([('a', 1), ('c', ('d', 3)), ('b', 2)])
    print(example)
    print(example.c)
    print(example['c'])
