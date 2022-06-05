from PIL import Image
import glob
i = 0
for dir in glob.glob('./VOCdevkit/VOC2012/JPEGImages/*'):
    i += 1
    img = Image.open(dir)
    m,n = img.size
    region = img.crop((m/2-112,n/2-112,m/2+112,n/2+112))
    region.save('./VOCdevkit/VOC2012/PNGImages/'+dir.split('/')[-1][:-3]+'png')
print(i) # 16134

import os
path = os.listdir('/data/xzn/ERR/ERR/VOCdevkit/VOC2012/PNGImages')
f = open('VOC2012_224_train_png.txt', 'w')
for line in path:
    f.write(line+'\n')
f.close()