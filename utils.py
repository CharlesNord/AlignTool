import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

img_root = '/mnt/data/cacd/CACD2000'
landmark_root = '/mnt/data/cacd/CACD_landmark/landmark'


def get_name():
    imgs = os.listdir(img_root)
    landmarks = [img[:-4] + '.landmark' for img in imgs]
    return landmarks, imgs


def read_landmark(lm_name):
    lm_path = os.path.join(landmark_root, lm_name)
    lm_array = np.loadtxt(lm_path)
    return lm_array


def read_image(im_name, img_root=img_root):
    im_path = os.path.join(img_root, im_name)
    I = cv2.imread(im_path)
    return I


def read_image_gray(im_name):
    im_path = os.path.join(img_root, im_name)
    I = cv2.imread(im_path, 0)
    return I


def show_landmarks(I, lm, scale=3):
    h, w, c = I.shape
    print(I.shape)
    new_h, new_w = h * scale, w * scale
    newI = cv2.resize(I, (new_w, new_h))
    print(newI.shape)
    lm = np.around(lm * scale, 0).astype('int')
    for i in range(16):
        x, y = lm[i]
        cv2.circle(newI, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(newI, '{}'.format(i), (x, y), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.5,
                    thickness=2, color=(255, 255, 0))
    return newI


def test_show_lm():
    landmarks, imgs = get_name()
    I = read_image(imgs[0])
    lm = read_landmark(landmarks[0])
    I = show_landmarks(I, lm)
    plt.imshow(I[:, :, ::-1])
    plt.show()


def test_read_file():
    landmarks, imgs = get_name()
    for i in range(len(landmarks)):
        landmark_path_i = os.path.join(landmark_root, landmarks[i])
        if not os.path.exists(landmark_path_i):
            print(landmark_path_i)
            continue
        else:
            if i % 500 == 0:
                print('{}/{}'.format(i, len(landmarks)))


if __name__ == '__main__':
    test_show_lm()
# test_read_file()
