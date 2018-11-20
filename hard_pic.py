import os
import cv2
from openface import AlignDlib
import dlib

# read_dir = '/mnt/data/MorphAlbum/morph'
# save_dir = '/mnt/data/MorphAlbum/lightcnn/'

save_dir = '/mnt/data/cacd/lightcnn_vs_color/'
read_dir = '/mnt/data/cacd/CACD_VS'

# img_list = ['0709_1.jpg', '2349_1.jpg', '2916_1.jpg', '3846_1.jpg', '0408_1.jpg', '0602_1.jpg', '2330_1.jpg',
#             '3998_1.jpg', '0523_1.jpg', '0723_0.jpg', '2329_0.jpg', '2307_1.jpg', '0019_0.jpg', '2803_0.jpg',
#             '2304_0.jpg', '3003_0.jpg', '3116_0.jpg', '0685_1.jpg', '3942_1.jpg', '3678_1.jpg', '1360_1.jpg',
#             '3823_0.jpg', '1190_0.jpg', '2341_0.jpg', '1033_1.jpg', '2302_1.jpg', '0457_1.jpg', '2141_1.jpg',
#             '0848_0.jpg', '2108_0.jpg', '0990_1.jpg', '2129_1.jpg', '3623_0.jpg', '0163_1.jpg', '2104_0.jpg',
#             '0657_0.jpg', '3643_0.jpg', '2832_0.jpg', '3693_0.jpg']
#
# img_list = ['034907_5M55.JPG', '150034_01M49.JPG']
img_list = ['3480_0.jpg', '3280_0.jpg']
# vs_root = '/mnt/data/lfw/Jeff_Feldman'
# save_dir = '/mnt/data/lfw128/Jeff_Feldman'
#
# img_list = ['Jeff_Feldman_0001.jpg']
rfPt = []


def mark_face(event, x, y, flags, param):
    global rfPt
    if event == cv2.EVENT_LBUTTONDOWN:
        rfPt = [x, y]
        print(rfPt)

    elif event == cv2.EVENT_LBUTTONUP:
        rfPt.extend([x, y])
        print(rfPt)

    return rfPt


cv2.namedWindow('image')
cv2.namedWindow('aligned')
cv2.setMouseCallback('image', mark_face)
align = AlignDlib()

for i in range(len(img_list)):
    print('Aligning {} ({}/{})'.format(img_list[i], i, len(img_list)))
    image = cv2.imread(os.path.join(read_dir, img_list[i]))
    # image = cv2.imread(os.path.join(vs_root, '0000_0.jpg'))[:, :, ::-1]
    cv2.imshow('image', image)
    key = cv2.waitKey(1) & 0xFF
    cv2.waitKey(0)

    if len(rfPt) != 0:
        bb = dlib.rectangle(*rfPt)
        out = align.align8(image, bb)
        cv2.imshow('aligned', out)
        # out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(save_dir, img_list[i]), out)
        print('Image {} aligned ({}/{})'.format(img_list[i], i, len(img_list)))
        cv2.waitKey(0)
        rfPt = []

print('Finished')
cv2.destroyAllWindows()
