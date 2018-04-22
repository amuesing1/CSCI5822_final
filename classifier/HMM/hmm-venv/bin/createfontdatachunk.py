#!/Users/mliuzzolino/Dropbox/CU_Boulder/Research/CAMP-SMC/interface_prototypes/interface_prototype_v1/backend-development/classifier/HMM/hmm-venv/bin/python
from __future__ import print_function
import base64
import os
import sys

if __name__ == "__main__":
    # create font data chunk for embedding
    font = "Tests/images/courB08"
    print("    f._load_pilfont_data(")
    print("         # %s" % os.path.basename(font))
    print("         BytesIO(base64.decodestring(b'''")
    base64.encode(open(font + ".pil", "rb"), sys.stdout)
    print("''')), Image.open(BytesIO(base64.decodestring(b'''")
    base64.encode(open(font + ".pbm", "rb"), sys.stdout)
    print("'''))))")
