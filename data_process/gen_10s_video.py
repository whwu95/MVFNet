"""cut videos"""
import argparse
import glob
import multiprocessing
import os
import os.path as osp
import subprocess

n_thread = 100


def parse_args():
    """针对wangxiaolong提供的K400 VAL 12s数据, 剪切成10s"""
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument('video_path', type=str,
                        help='root directory for the input videos')
    parser.add_argument('out_path', type=str,
                        help='root directory for the out videos')
    parser.add_argument('--level', type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()
    return args


def cut_video(tup):
    """cut video"""
    src, dest = tup
    folder, name = osp.split(dest)

    if not osp.exists(folder):
        os.system('mkdir -p {}'.format(folder))

    cmd1 = 'ffprobe -i \"{}\"  -show_entries \
        format=duration -v quiet -of csv="p=0"'.format(src)
    duration = subprocess.check_output(
        cmd1, shell=True).decode('utf-8').strip()
    if round(float(duration)) > 10:
        cmd = 'ffmpeg -y -loglevel error \
            -ss 00:01 -t 10 -i  \"{}\" -threads 1 \"{}\"'.format(src, dest)
        retcode = subprocess.call(cmd, shell=True)
        if retcode == 1:  # if sth wrong
            os.system('cp  \"{}\"  \"{}\"'.format(src, dest))
    else:
        os.system('cp  \"{}\"  \"{}\"'.format(src, dest))
    return


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

    # 1: Use tqdm for progress bar
    from tqdm import tqdm
    for _ in tqdm(pool.imap_unordered(cut_video, vid_list),
                  total=len(vid_list)):
        pass
    # 2: Without progress Bar
    # pool.map(cut_video, vid_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
