"""Generate list of train/val set for multiple video datasets"""
import argparse
import json
import os
import os.path as osp
import shutil


def parse_args():
    """Build file label list"""
    parser = argparse.ArgumentParser(description='Build file label list')
    parser.add_argument('data_path', type=str,
                        help='root directory for the dataset')
    parser.add_argument('dataset', type=str, choices=[
                        'ucf101', 'hmdb51',
                        'kinetics400', 'kinetics600', 'kinetics700',
                        'sthv1', 'sthv2'],
                        help='name of the dataset')
    parser.add_argument('--ann_root', type=str, default='annotation')
    parser.add_argument('--out_root', type=str, default='../datalist')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'val'])
    parser.add_argument('--level', type=int, default=2, choices=[1, 2])
    parser.add_argument('--source', type=str, default='rgb',
                        choices=['rgb', 'flow', 'video'])
    parser.add_argument('--split', type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    return args


def parse_label_file(file):
    """parse annotation file"""
    categories = []
    with open(file) as f:
        if file.endswith('json'):
            data = json.load(f)
            for i, (cat, idx) in enumerate(data.items()):
                assert i == int(idx)  # make sure the rank is right
                categories.append(cat)
        elif 'kinetics' in file:
            lines = f.readlines()
            categories = [c.strip().replace(' ', '_').replace(
                '"', '').replace('(', '').replace(
                ')', '').replace("'", '') for c in lines]
        else:
            lines = f.readlines()
            categories = [line.rstrip() for line in lines]

    if 'sthv1' in file:
        categories = sorted(categories)
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    return dict_categories


def gen_sth_label(data_path, ann_path, out_path, source='rgb'):
    """
    sthv1: csv
    sthv2: json
    """

    def parse_video_file(file):
        """parse sth-sth v1 & v2 annotation"""
        folders = []
        idx_categories = []
        if file.endswith('json'):
            with open(file) as f:
                data = json.load(f)
            for item in data:
                folders.append(item['id'])
                if 'test' not in filename_input:
                    idx_categories.append(
                        dict_categories[
                            item['template'].replace(
                                '[', '').replace(']', '')])
                else:
                    idx_categories.append(0)
        elif file.endswith('csv'):
            with open(file) as f:
                lines = f.readlines()
            for line in lines:
                items = line.rstrip().split(';')
                folders.append(items[0])
                idx_categories.append(dict_categories[items[1]])
        return folders, idx_categories

    if 'sthv1' in ann_path:
        dataset_name = 'something-something-v1'
        label_file = osp.join(ann_path, '%s-labels.csv' % dataset_name)
        files_input = [osp.join(ann_path, '%s-validation.csv' % dataset_name),
                       osp.join(ann_path, '%s-train.csv' % dataset_name)]
        files_output = [osp.join(out_path, 'val_%s.txt' % source),
                        osp.join(out_path, 'train_%s.txt' % source)]

    elif 'sthv2' in ann_path:
        dataset_name = 'something-something-v2'
        label_file = osp.join(ann_path, '%s-labels.json' % dataset_name)
        files_input = [osp.join(ann_path, '%s-validation.json' % dataset_name),
                       osp.join(ann_path, '%s-train.json' % dataset_name),
                       osp.join(ann_path, '%s-test.json' % dataset_name)]
        files_output = [osp.join(out_path, 'val_%s.txt' % source),
                        osp.join(out_path, 'train_%s.txt' % source),
                        osp.join(out_path, 'test_%s.txt' % source)]

    dict_categories = parse_label_file(label_file)

    for (filename_input, filename_output) in zip(
            files_input, files_output):

        folders, idx_categories = parse_video_file(filename_input)

        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join(
                data_path, curFolder))
            if source == 'flow':
                dir_files = [x for x in dir_files if 'flow_x' in x]
            output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))


def gen_kinetics_label(data_path, ann_path, out_path, level=1,
                       source='rgb', phase='train', start_end=False):
    """generate kinetics datalist"""
    if '400' in ann_path:
        num_class = 400
    elif '600' in ann_path:
        num_class = 600
    elif '700' in ann_path:
        num_class = 700

    lable_file = osp.join(ann_path, 'kinetics-%d_label_map.txt' % num_class)
    file_input = osp.join(ann_path, 'kinetics-%d_%s.csv' % (num_class, phase))
    file_out = 'kinetics_%s_%s.txt' % (source, phase)

    dict_categories = parse_label_file(lable_file)
    assert len(dict_categories.keys()) == num_class
    # print(dict_categories)

    # rename class folder
    if level == 2:
        classes = os.listdir(data_path)
        for name in classes:
            dst = name.strip().replace(' ', '_').replace('"', '').replace(
                '(', '').replace(')', '').replace("'", '')
            if name != dst:
                shutil.move(
                    osp.join(data_path, name), osp.join(data_path, dst))

    count_cat = {k: 0 for k in dict_categories.keys()}
    with open(file_input) as f:
        lines = f.readlines()[1:]
    folders = []
    categories_list = []

    for line in lines:
        items = line.rstrip().split(',')
        folders.append(
            items[1] + '_' + '%06d' % int(
                items[2]) + '_' + '%06d' % int(items[3]))
        this_catergory = items[0].replace(' ', '_').replace(
            '"', '').replace('(', '').replace(')', '').replace("'", '')
        categories_list.append(this_catergory)
        count_cat[this_catergory] += 1
    # print(max(count_cat.values()))

    assert len(categories_list) == len(folders)
    missing_folders = []
    output = []

    for i in range(len(folders)):
        # without [:11], the foldername should have _start_end
        curFolder = folders[i] if start_end else folders[i][:11]
        category = categories_list[i]
        curIDX = dict_categories[category]
        if level == 1:
            sub_dir = curFolder
        elif level == 2:
            sub_dir = osp.join(category, curFolder)
        # counting the number of frames in each video folders
        img_dir = osp.join(data_path, sub_dir)

        if source == 'video':
            import glob
            vid_path = glob.glob(img_dir + '*')
            if len(vid_path) == 0:
                missing_folders.append(img_dir)
            else:
                vid_name = osp.split(vid_path[0])[-1]
                output.append('%s %d' % (os.path.join(
                    category, vid_name), curIDX))
        else:
            if not os.path.exists(img_dir):
                missing_folders.append(img_dir)
                # print(missing_folders)
            else:
                dir_files = os.listdir(img_dir)
                if source == 'flow':
                    dir_files = [x for x in dir_files if 'flow_x' in x]

                output.append('%s %d %d' % (sub_dir, len(dir_files), curIDX))

        print('%d/%d, missing %d' % (i, len(folders), len(missing_folders)))
    with open(os.path.join(out_path, file_out), 'w') as f:
        f.write('\n'.join(output))
    with open(os.path.join(out_path, 'missing_' + file_out), 'w') as f:
        f.write('\n'.join(missing_folders))


def gen_label(data_path, ann_path, out_path, source, split):
    """generate datalist for UCF101 and HMDB51"""
    label_file = osp.join(ann_path, 'category.txt')
    files_input = [osp.join(ann_path, 'trainlist0%d.txt' % split),
                   osp.join(ann_path, 'testlist0%d.txt' % split)]
    files_output = [osp.join(out_path, 'train_%s_split_%d.txt' % (
                        source, split)),
                    osp.join(out_path, 'val_%s_split_%d.txt' % (
                        source, split))]

    dict_categories = parse_label_file(label_file)

    for (filename_input, filename_output) in zip(
            files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        categories_list = []
        for line in lines:
            label, name = osp.split(line.rstrip())
            folders.append(name.split('.')[0])
            categories_list.append(label)

        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            category = categories_list[i]
            curIDX = dict_categories[category]
            img_dir = osp.join(data_path, category, curFolder)
            if source == 'video':
                import glob
                vid_path = glob.glob(img_dir + '*')[0]
                vid_name = osp.split(vid_path)[-1]
                output.append('%s %d' % (os.path.join(
                    category, vid_name), curIDX))
            else:
                dir_files = os.listdir(img_dir)
                if source == 'flow':
                    dir_files = [x for x in dir_files if 'flow_x' in x]
                output.append('%s %d %d' % (
                    osp.join(category, curFolder), len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))


def main():
    """main"""
    args = parse_args()
    dataset = args.dataset
    ann_path = osp.join(args.ann_root, args.dataset)
    out_path = osp.join(args.out_root, args.dataset)
    if not osp.exists(out_path):
        os.system('mkdir -p {}'.format(out_path))
    if 'sth' in dataset:
        gen_sth_label(args.data_path, ann_path, out_path, args.source)
    elif 'kinetics' in dataset:
        gen_kinetics_label(args.data_path, ann_path, out_path,
                           args.level, args.source, args.phase)
    elif dataset in ['ucf101', 'hmdb51']:
        gen_label(args.data_path, ann_path, out_path,
                  args.source, args.split)


if __name__ == "__main__":
    main()
