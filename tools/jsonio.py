import os
import json

def read_json(f:os.PathLike):
    ret = None
    with open(f, "r") as fp:
        ret = json.load(fp)
    return ret