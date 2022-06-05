from PIL import Image
import glob
import os
items = glob.glob('./myuaresults/*')
cnt = 0
for it in items:
    picname = it.split("/")[-1]
    pic_input = Image.open("./results/" + picname + "/m_input.png")
    pic_errnet = Image.open("./results/" + picname + "/errnet.png")
    pic_vgg = Image.open("./uaresults/" + picname + "/errnet.png")
    pic_cobi = Image.open("./myuaresults/" + picname + "/errnet.png")
    if not os.path.exists("./compare/"+ picname):
        os.mkdir("./compare/"+ picname)
    pic_input.save("./compare_model_output/"+ picname + '/input.png')
    pic_errnet.save("./compare_model_output/"+ picname + '/errnet.png')
    pic_vgg.save("./compare_model_output/"+ picname + '/vgg.png')
    pic_cobi.save("./compare_model_output/"+ picname + '/cobi.png')
    print(picname)
