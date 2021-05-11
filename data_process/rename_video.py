import argparse
import glob
import multiprocessing
import os
import os.path as osp

n_thread = 100


def parse_args():
    parser = argparse.ArgumentParser(description='Rename kinetics label')
    parser.add_argument('video_path', type=str,
                        help='root directory for the input videos')
    parser.add_argument('--level', type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()
    return args


def rename_video(src):
    folder, video = osp.split(src)
    tmp = video.split('.')
    new_list = [tmp[0][:11]]
    new_list.extend(tmp[1:])
    new_video = '.'.join(new_list)
    video_path = osp.join(folder, new_video)
    if video_path != src:
        os.system('mv  \"{}\"  \"{}\"'.format(src, video_path))
    return


def main():
    args = parse_args()

    if args.level == 1:
        vid_list = glob.glob(osp.join(args.video_path, '*'))

        # ['root/xxx.mp4']
    elif args.level == 2:
        vid_list = glob.glob(osp.join(args.video_path, '*', '*'))

        # ['root/class/xxx.mp4']
    elif args.level == 3:
        vid_list = glob.glob(osp.join(args.video_path, '*', '*', '*'))
        # ['root/class/sub/xxx.mp4']

    pool = multiprocessing.Pool(n_thread)

    # 1: Use tqdm for progress bar
    from tqdm import tqdm
    for _ in tqdm(pool.imap_unordered(rename_video, vid_list),
                  total=len(vid_list)):
        pass
    # 2: Without progress Bar
    # pool.map(cut_video, vid_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
