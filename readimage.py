import json
from PIL import Image
import sys
import numpy as np

im = Image.fromarray(np.uint8(np.array(json.load(open(sys.argv[1]))) * 255))
im.show()