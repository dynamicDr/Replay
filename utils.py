import numpy as np
import uuid

def get_MAC():
    mac = uuid.getnode()
    return mac

if __name__ == '__main__':
    print(get_MAC())
