from PIL import Image

tes = Image.open("./people_jpeg/353001.png")
print(max(list(tes.getdata())))