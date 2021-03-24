"""resize video
"""
import argparse
import glob
import multiprocessing
import os
import os.path as osp
import subprocess
from functools import partial

n_thread = 100


def parse_args():
    """[summary]

    Returns:
        [type]: [description]
    """
    parser = argparse.ArgumentParser(description='Rescale videos')
    parser.add_argument('video_path', type=str,
                        help='root directory for the input videos')
    parser.add_argument('out_path', type=str,
                        help='root directory for the out videos')
    parser.add_argument('--level', type=int, default=2, choices=[1, 2, 3],
                        help='the number of level for folders, 1 for sth-sth,\
                        2 for kinetics, 3 for ucf101/hmdb51')
    parser.add_argument('-se', '--short_edge', type=int, default=256)
    args = parser.parse_args()
    return args


def vid_resize(tup, se):
    """[summary]

    Args:
        tup ([type]): [description]
        se ([type]): [description]
    """
    src, dest = tup
    folder, name = osp.split(dest)  # ['root/zumba'], ['xxx.mp4']

    if not osp.exists(folder):
        os.system('mkdir -p {}'.format(folder))

    cmd1 = 'ffprobe -v error -select_streams v:0 -show_entries \
        stream=width,height -of csv=s=x:p=0 \"{}\"'.format(src)
    width, height = subprocess.check_output(
        cmd1, shell=True).decode('utf-8').split('x')
    if int(width) > int(height):
        os.system('ffmpeg -y -loglevel panic -i {} -threads 1 \
            -filter:v scale=-2:{} -q:v 1 -c:a copy {}'.format(src, se, dest))
    else:
        os.system('ffmpeg -y -loglevel panic -i {} -threads 1 \
            -filter:v scale={}:-2 -q:v 1 -c:a copy {}'.format(src, se, dest))


def main():
    """main
    """
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
    worker = partial(vid_resize, se=args.short_edge)
    # 1: Use tqdm for progress bar
    from tqdm import tqdm
    for _ in tqdm(pool.imap_unordered(worker, vid_list),
                  total=len(vid_list)):
        pass
    # 2: Without progress Bar
    # pool.map(vid_resize, vid_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
