from PIL import Image
import sys
import math

img = Image.open(sys.argv[1])
img_rgb = img.convert("RGB")
pixels = img_rgb.load()

for i in range(img_rgb.size[0]):
    for j in range(img_rgb.size[1]):
    	r, g, b = pixels[i, j]
    	pixels[i, j] = (int(r/2), int(g/2), int(b/2))
img_rgb.save("Q2.png", "PNG")


