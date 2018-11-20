import os
from openface import AlignDlib
from glob import glob
import cv2

root_lfw = '/mnt/data/lfw'
save_dir = '/mnt/data/lfw128'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

align = AlignDlib()


def obj_pth():
    src_pths = os.path.join(root_lfw, '*')
    src_pths = glob(src_pths)
    objs = os.listdir(root_lfw)
    dst_pths = [os.path.join(save_dir, name) for name in objs]
    for pth in dst_pths:
        if not os.path.exists(pth):
            os.mkdir(pth)
    return src_pths, dst_pths


def get_align(pth):
    I = cv2.imread(pth)[:, :, ::-1]
    out = align.align7(I)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return out


def main():
    src_pths, dst_pths = obj_pth()
    hard_imgs = []
    for i in range(len(src_pths)):
        src = src_pths[i]
        dst = dst_pths[i]
        if src == '/mnt/data/lfw/pairs.txt':
            continue
        objs = os.listdir(src)
        if len(objs) == len(os.listdir(dst)):
            continue
        for obj in objs:
            src_img = os.path.join(src, obj)
            dst_img = os.path.join(dst, obj)
            try:
                out = get_align(src_img)
                cv2.imwrite(dst_img, out)
            except:
                hard_imgs.append(src_img)
                print(src_img)

        if i % 50 == 0:
            print('Aligned {} ({}) objects'.format(i, len(src_pths)))


if __name__ == '__main__':
    main()
