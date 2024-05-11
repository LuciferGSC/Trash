import os

if __name__ == '__main__':
    path = './ori'
    list = []
    for c in os.listdir(path):
        print(c)
        list.append(c)
    print(list)
