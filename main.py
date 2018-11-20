from openface import AlignDlib
from utils import read_landmark, read_image_gray, get_name
import os
import cv2
from multiprocessing import Pool
from tqdm import tqdm

save_dir = '/mnt/data/cacd/lightcnn'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

landmarks, images = get_name()
align = AlignDlib()


def work(i):
    lm_name = landmarks[i]
    im_name = images[i]
    image = read_image_gray(im_name)
    lm = read_landmark(lm_name)
    out = align.align9(image, lm)
    cv2.imwrite(os.path.join(save_dir, im_name), out)


if __name__ == '__main__':
    pool = Pool(8)
    for _ in tqdm(pool.imap(work, range(len(landmarks))), total=len(landmarks)):
        pass

    pool.close()
    pool.join()
