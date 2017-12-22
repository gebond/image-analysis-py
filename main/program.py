from PIL import Image

from main.classes import ImageBuilder, Math, ImageUtils

# set image name
IMAGE = 'car3'

VERTICAL_THRESHOLD = 1.5
HORIZONTAL_THRESHOLD = 1.7
SLOOP_THRESHOLD = 2.5
NOISE_THRESHOLD = 5.0

# 0 - 4 mask number
MASK = 2

print("Started with: " + IMAGE + '.png')
im = Image.open('../images/' + IMAGE + '.png')
rgb_im = im.convert('RGB')

builder = ImageBuilder()

average = builder\
    .with_img_average(rgb_im)\
    .normalize()\
    .laplasian()\
    .vertical_horizontal_mask(VERTICAL_THRESHOLD, HORIZONTAL_THRESHOLD)\
    .build()

ImageUtils.save_matrix_as_image(IMAGE, 'av-color-laplasian-vert-hor-mask', average, un_norm=True)
print("Ok. Finished")

red_color = builder\
    .with_img_channel(rgb_im, 'RED')\
    .normalize()\
    .laplasian()\
    .vertical_horizontal_mask(VERTICAL_THRESHOLD, HORIZONTAL_THRESHOLD)\
    .build()

ImageUtils.save_matrix_as_image(IMAGE, 'red-color-laplasian-vert-hor-mask', red_color, un_norm=True)
print("Ok. Finished")

laplasian = builder\
    .with_img_channel(rgb_im, 'RED')\
    .normalize()\
    .laplasian()\
    .build()

ImageUtils.save_matrix_as_image(IMAGE, 'red-color-laplasian', laplasian, un_norm=True)
print("Ok. Finished")
