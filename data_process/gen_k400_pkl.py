""" 
example command line: python gen_k400_pkl.py video_dir  out_dir
pkl will save to afs
"""
import sys
import _pickle as cPickle
import argparse
import glob
import multiprocessing
import os
import os.path as osp
import subprocess
from functools import partial



n_thread = 50


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='Rescale videos')
    parser.add_argument('video_path', type=str,
                        help='root directory for the input videos')
    parser.add_argument('out_path', type=str,
                        help='root directory for the out pkl')
    parser.add_argument('--level', type=int, default=2, choices=[1, 2, 3],
                        help='the number of level for folders')
    parser.add_argument('-fps', type=int, default=30)
    args = parser.parse_args()
    return args


def vid2pkl(tup, fps=None, prefix='img_%05d.jpg'):
    """video2pkl"""
    src, dest = tup
    folder, vid_name = osp.split(dest)
    video_name = vid_name.split('.')[0]
    video_folder = osp.join(folder, video_name)

    output_pkl = video_folder + '.pkl'
    try:
        if osp.exists(video_folder):
            if not osp.exists(
                    osp.join(video_folder, prefix.format(1))):
                subprocess.call(
                    'rm -r \"{}\"'.format(video_folder), shell=True)
                print('remove {}'.format(video_folder))
                os.system('mkdir -p {}'.format(video_folder))
            else:
                print('*** convert has been done: {}'.
                      format(video_folder))
                return
        else:
            os.system('mkdir -p {}'.format(video_folder))
    except Exception:
        print(video_folder)
        return

    cmd1 = 'ffprobe -v error -select_streams v:0 -show_entries \
        stream=width,height -of csv=s=x:p=0 \"{}\"'.format(src)
    try:
        w, h = subprocess.check_output(
            cmd1, shell=True).decode('utf-8').split('x')
    except ValueError:
        print(video_folder)
        return

    if fps is None:
        cmd = './ffmpeg -i \"{}\"  -threads 1 -q:v 1 \
            \"{}/{}\"'.format(src, video_folder, prefix)
    else:
        cmd = './ffmpeg -i \"{}\"  -threads 1 -q:v 1 -r {} \
            \"{}/{}\"'.format(src, fps, video_folder, prefix)

    # decode image
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    images = sorted(glob.glob(video_folder + '/*.jpg'))
    # print(images)
    ims = []
    for img in images:
        # note: python3 open with 'rb'
        with open(img, 'rb') as f:
            ims.append(f.read())
    with open(output_pkl, 'wb') as f:
        cPickle.dump(ims, f, -1)
    



    os.system('rm -rf %s' % video_folder)
    # os.system('rm %s' % output_pkl)


def main():
    """main"""
    args = parse_args()

    if args.level == 1:
        src_list = glob.glob(osp.join(args.video_path, '*'))
        dest_list = [osp.join(args.out_path, osp.split(vid)[-1])
                     for vid in src_list]
        # ['root/xxx.mp4']
    elif args.level == 2:
        src_list = glob.glob(osp.join(args.video_path, '*', '*'))
        dest_list = [osp.join(
            args.out_path, vid.split('/')[-2], osp.split(vid)[-1])
            for vid in src_list]
        # ['root/class/xxx.mp4']
    elif args.level == 3:
        src_list = glob.glob(osp.join(args.video_path, '*', '*', '*'))
        dest_list = [osp.join(
            args.out_path, vid.split('/')[-3],
            vid.split('/')[-2], osp.split(vid)[-1])
            for vid in src_list]
        # ['root/class/sub/xxx.mp4']

    vid_list = list(zip(src_list, dest_list))

    pool = multiprocessing.Pool(n_thread)
    worker = partial(vid2pkl, fps=args.fps)
    # 1: Use tqdm for progress bar
    from tqdm import tqdm
    for _ in tqdm(pool.imap_unordered(worker, vid_list),
                  total=len(vid_list)):
        pass

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()