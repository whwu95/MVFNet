"""mv file using multiprocessing"""
import argparse
import glob
import multiprocessing
import os
import os.path as osp
import shutil
from functools import partial

n_thread = 50


def parse_args():
    """[summary]

    Returns:
        [type]: [description]
    """
    parser = argparse.ArgumentParser(description='Rename kinetics label')
    parser.add_argument('src', type=str,
                        help='root directory for the input videos')
    parser.add_argument('dst', type=str,
                        help='target directory for the input videos')
    parser.add_argument('--level', type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    return args


def mv_folder(src, dst):
    """[summary]

    Args:
        src ([type]): [description]
        dst ([type]): [description]
    """
    file, subfile = osp.split(src)
    file_name = file.split('/')[-1]
    dst_file = osp.join(dst, file_name)
    if not os.path.exists(dst_file):
        os.makedirs(dst_file)
    new_path = osp.join(dst_file, subfile)
    shutil.move(src, new_path)
    return


def main():
    """main
    """
    args = parse_args()

    if args.level == 1:
        vid_list = glob.glob(osp.join(args.src, '*'))

        # ['root/xxx.mp4']
    elif args.level == 2:
        vid_list = glob.glob(osp.join(args.src, '*', '*'))

        # ['root/class/xxx.mp4']
    elif args.level == 3:
        vid_list = glob.glob(osp.join(args.src, '*', '*', '*'))
        # ['root/class/sub/xxx.mp4']

    pool = multiprocessing.Pool(n_thread)
    worker = partial(mv_folder, dst=args.dst)
    # 1: Use tqdm for progress bar
    from tqdm import tqdm
    for _ in tqdm(pool.imap_unordered(worker, vid_list),
                  total=len(vid_list)):
        pass
    # 2: Without progress Bar
    # pool.map(cut_video, vid_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
