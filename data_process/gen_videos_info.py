"""generate video info: resolution, duration"""
import argparse
import glob
import os.path as osp
import subprocess


def parse_args():
    """parse"""
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument('video_path', type=str,
                        help='root directory for the frames')
    parser.add_argument('--out', type=str, default='video_duration.txt',
                        help='out path for the list')
    parser.add_argument('--info', type=str, default='resolution',
                        choices=['resolution', 'duration'])
    parser.add_argument('--level', type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()
    return args


def get_duration(vid):
    """[summary]

    Args:
        vid ([type]): [description]

    Returns:
        [type]: [description]
    """
    cmd1 = 'ffprobe -i \"{}\"  -show_entries \
        format=duration -v quiet -of csv="p=0"'.format(vid)
    duration = subprocess.check_output(cmd1, shell=True).decode('utf-8')
    return duration


def get_resolution(vid):
    """[summary]

    Args:
        vid ([type]): [description]

    Returns:
        [type]: [description]
    """
    cmd1 = 'ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height -of csv=s=x:p=0 \"{}\"'.format(
        vid)
    widthxheight = subprocess.check_output(
        cmd1, shell=True).decode('utf-8')
    return widthxheight


def main():
    """[main]
    """
    args = parse_args()

    if args.level == 1:
        video_list = glob.glob(osp.join(args.video_path, '*'))
        # ['root/xxx.mp4']
    elif args.level == 2:
        video_list = glob.glob(osp.join(args.video_path, '*', '*'))
        # ['root/class/xxx.mp4']
    elif args.level == 3:
        video_list = glob.glob(osp.join(args.video_path, '*', '*', '*'))
        # ['root/class/sub/xxx.mp4']
    with open(args.out, 'w+') as f:
        for i in range(len(video_list)):
            vid = video_list[i]
            name = vid.split('/')[-1].split('.')[0][:11]
            if i != 0 and i % 500 == 0:
                print('%d videos have done' % i)
            if args.info == 'resolution':
                info = get_resolution(vid)
            elif args.info == 'duration':
                info = get_duration(vid)
            f.write(name + ' ' + str(info))


if __name__ == "__main__":
    main()
