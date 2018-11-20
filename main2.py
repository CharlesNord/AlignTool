from openface import AlignDlib
import os
import cv2
from multiprocessing import Pool
from tqdm import tqdm

save_dir = '/mnt/data/cacd/lightcnn_vs_color/'
img_root = '/mnt/data/cacd/CACD_VS'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


images = os.listdir(img_root)
align = AlignDlib()


def work(i):
    im_name = images[i]
    image = cv2.imread(os.path.join(img_root,im_name))
    out = align.align7(image)
    #out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(save_dir, im_name), out)


# if __name__ == '__main__':
#     pool = Pool(8)
#     for _ in tqdm(pool.imap(work, range(len(images))), total=len(images)):
#         pass
#
#     pool.close()
#     pool.join()

exceptions = []

for i in range(len(images)):
    try:
        work(i)
        if i% 50==0:
            print('{}/{}'.format(i, len(images)))
    except:
        print(images[i])
        exceptions.append(images[i])

print(exceptions)

