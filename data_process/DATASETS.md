## Video Dataset Preparation

Your are expected to prepare data for [Kinetics-400](https://deepmind.com/research/open-source/kinetics) dataset, [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads), [Something-something-v1](https://20bn.com/datasets/something-something/v1) and [Something-something-v2](https://20bn.com/datasets/something-something) datasets.


### Official Downloader
* Kinetics-400: Download the videos via the official [scripts](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). We download the copy from the [Nonlocal Network](https://github.com/facebookresearch/video-nonlocal-net/blob/master/DATASET.md).
* UCF101: Download the videos via the [official website](https://www.crcv.ucf.edu/data/UCF101.php)
* HMDB51: Download the videos via the [official website](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)
* Something-Something V1: Download the `video frames` via the [official website](https://20bn.com/datasets/something-something/v1)
* Something-Something V2: Download the videos via the [official website](https://20bn.com/datasets/something-something/v2)



## Dataset Processing
This repo supports **raw frame** format of videos. Therefore, you can extract frames from raw videos. Also, we support a fast **VideoLoader**, which can be found at [Fast PyAVDecode](https://github.com/whwu95/MVFNet/blob/83dcb881c2552524d3f861a38133fee44449a979/codes/datasets/pipelines/loading.py#L135).


### Resize original videos (Optional)

```Shell
# Use FFmpeg to rescale the videos to short edge =256 pixels. 
python video_resize.py ROOT_PATH OUT_PATH --level 2 -se 256
```
### Extract raw frames
Extract frames for training and validation, use --lib to choose opencv or ffmpeg

```Shell
# For Kinetics400
python video2image.py ROOT_PATH OUT_PATH --level 2  --lib ffmpeg -fps 30

# For Sth-Sth v2
python video2image.py ROOT_PATH OUT_PATH --level 1 -se 256 --lib opencv --prefix %06d.jpg

# For UCF101
python video2image.py ROOT_PATH OUT_PATH --level 2 --lib opencv --prefix image_%04d.jpg

# For HMDB51
python video2image.py ROOT_PATH OUT_PATH --level 2 --lib opencv --prefix image_%06d.jpg
```

### Prepare annotations 
Prepare label list, [video_name, #frames, label] for each row, and save them in `datalist` folder.

```Shell
# For sthv1/v2
python gen_label.py IMAGE_ROOT sthv1

# kinetics400/600/700
python gen_label.py IMAGE_ROOT kinetics400 --phase train  --level 1
python gen_label.py VIDEO_ROOT kinetics400 --phase train  --level 2 --source video

# For ucf101/hmdb51
python gen_label.py IMAGE_ROOT ucf101  --split 1
python gen_label.py VIDEO_ROOT ucf101  --split 1 --source video
python gen_label.py FLOW_ROOT ucf101  --split 1 --source flow
````


