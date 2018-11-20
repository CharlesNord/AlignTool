from openface import AlignDlib
from utils import read_image
import os
import cv2
from multiprocessing import Pool
from tqdm import tqdm

read_dir = '/mnt/data/MorphAlbum/morph'
save_dir = '/mnt/data/MorphAlbum/lightcnn'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

images = os.listdir(read_dir)
align = AlignDlib()


def work(i):
    im_name = images[i]
    image = read_image(im_name, img_root=read_dir)
    out = align.align7(image)
    cv2.imwrite(os.path.join(save_dir, im_name), out)


if __name__ == '__main__':
    # pool = Pool(8)
    # for _ in tqdm(pool.imap(work, range(len(images))), total=len(images)):
    #     pass
    #
    # pool.close()
    # pool.join()
    exceptions = []
    for i in range(len(images)):
        try:
            work(i)
            if i % 50 == 0:
                print('{}/{}'.format(i, len(images)))
        except:
            print(images[i])
            exceptions.append(images[i])

    print(exceptions)