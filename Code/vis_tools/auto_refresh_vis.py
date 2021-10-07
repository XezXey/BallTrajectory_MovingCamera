import os
import pathlib
p = pathlib.Path('../reconstructed/')
b4 = p.stat()
while True:
    if (b4 != p.stat()):
        break
print(p.stat())