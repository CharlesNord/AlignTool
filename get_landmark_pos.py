# revers engineering of the aligned face images
import dlib
import cv2
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt

aligned_root = '/mnt/data/lfw_lightcnn'
imgs = glob(os.path.join(aligned_root, '*', '*.bmp'))

facePredictor = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(facePredictor)
bb = dlib.rectangle(0, 0, 128, 128)


def get_landmark(pth):
    I = cv2.imread(pth, 0)
    points = predictor(I, bb)
    points = np.array(list(map(lambda p: (p.x, p.y), points.parts())))

    left_eye = points[range(36, 42)].mean(0)
    right_eye = points[range(42, 48)].mean(0)
    nose = points[30]
    left_mouth = points[48]
    right_mouth = points[54]
    return left_eye, right_eye, nose, left_mouth, right_mouth


def check(pth, landmarks):
    I = cv2.imread(pth)
    for (i, (x, y)) in enumerate(landmarks):
        cv2.circle(I, (int(x), int(y)), 1, (0, 0, 255), -1)
    return I


def main():
    left_eye = []
    right_eye = []
    nose = []
    left_mouth = []
    right_mouth = []
    for i in range(len(imgs)):
        if (i+1)%50 == 0:
            print('{}/({}) images read'.format(i+1, len(imgs)))
        pth = imgs[i]
        le, re, n, lm, rm = get_landmark(pth)
        left_eye.append(le)
        right_eye.append(re)
        nose.append(n)
        left_mouth.append(lm)
        right_mouth.append(rm)

    left_eye_mean = np.array(left_eye).mean(0)
    right_eye_mean = np.array(right_eye).mean(0)
    left_mouth_mean = np.array(left_mouth).mean(0)
    right_mouth_mean = np.array(right_mouth).mean(0)
    nose_mean = np.array(nose).mean(0)

    I = check(imgs[1], (left_eye_mean,
                        right_eye_mean,
                        nose_mean,
                        left_mouth_mean,
                        right_mouth_mean))
    plt.imshow(I)
    plt.show()
    print('left_eye_mean: ',left_eye_mean)
    print('right_eye_mean: ',right_eye_mean)
    print('nose_mean: ',nose_mean)
    print('left_mouth_mean: ', left_mouth_mean)
    print('right_mouth_mean: ', right_mouth_mean)


main()
