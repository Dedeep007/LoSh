import argparse
import ruamel.yaml
import os
from torch.utils.data import DataLoader, DistributedSampler


import json
import torch
import numpy as np
from torchvision.io import read_video
import h5py
from torch.utils.data import Dataset
import torch.distributed as dist
import torchvision.transforms.functional as F
import pandas
from os import path
from glob import glob
from tqdm import tqdm

import datasets.transforms as T
from pycocotools.mask import encode, area
from misc import nested_tensor_from_videos_list
from datasets.a2d_sentences.create_gt_in_coco_format import create_a2d_sentences_ground_truth_test_annotations

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
nltk.download('tagsets')

def get_image_id(video_id, frame_idx, ref_instance_a2d_id):
    image_id = f'v_{video_id}_f_{frame_idx}_i_{ref_instance_a2d_id}'
    return image_id

class A2DSentencesDataset(Dataset):
    """
    A Torch dataset for A2D-Sentences.
    For more information check out: https://kgavrilyuk.github.io/publication/actor_action/ or the original paper at:
    https://arxiv.org/abs/1803.07485
    """
    def __init__(self, subset_type: str = 'train', dataset_path: str = './a2d_sentences', window_size=8,
                 dataset_coco_gt_format_path=None, distributed=False, **kwargs):
        super(A2DSentencesDataset, self).__init__()
        assert subset_type in ['train', 'test'], 'error, unsupported dataset subset type. supported: train, test'
        self.subset_type = subset_type
        self.mask_annotations_dir = path.join(dataset_path, 'text_annotations/a2d_annotation_with_instances')
        self.videos_dir = path.join(dataset_path, 'Release/clips320H')
        self.text_annotations = A2DSentencesDataset.get_text_annotations(dataset_path, subset_type, distributed)
        #print("Text Annotations",self.text_annotations)
        self.window_size = window_size
        self.transforms = A2dSentencesTransforms(subset_type, **kwargs)
        self.collator = Collator()
        # create ground-truth test annotations for the evaluation process if necessary:
        if subset_type == 'test' and not path.exists(dataset_coco_gt_format_path):
            if (distributed and dist.get_rank() == 0) or not distributed:
                create_a2d_sentences_ground_truth_test_annotations()
            if distributed:
                dist.barrier()

    #Changes made by Dedeep.v.: Added a function to generate short text query
    @staticmethod
    def generate_short_text_query(query):
        words = word_tokenize(query) 
        pos_tags = pos_tag(words)  
        verb_index = next((i for i, (_, tag) in enumerate(pos_tags) if tag.startswith('VB')), len(pos_tags))
        short_text = " ".join(words[:verb_index])
        return short_text
    
    #Changes made by Dedeep.v.: Implemented the generate short text query function in this function
    @staticmethod
    def get_text_annotations(root_path, subset, distributed):
        saved_annotations_file_path = f'./datasets/a2d_sentences/a2d_sentences_single_frame_{subset}_annotations.json'
        # if path.exists(saved_annotations_file_path):
        #     with open(saved_annotations_file_path, 'r') as f:
        #         print("in first if")
        #         text_annotations_by_frame = [tuple(a) for a in json.load(f)]
        #         return text_annotations_by_frame
        if (distributed and dist.get_rank() == 0) or not distributed:
            print(f'building a2d sentences {subset} text annotations...')
            # without 'header == None' pandas will ignore the first sample...
            a2d_data_info = pandas.read_csv(path.join(root_path, 'Release/videoset.csv'), header=None)
            assert len(a2d_data_info) == 3782, f'error: a2d videoset.csv file is missing one or more samples.'
            # 'vid', 'label', 'start_time', 'end_time', 'height', 'width', 'total_frames', 'annotated_frames', 'subset'
            a2d_data_info.columns = ['vid', '', '', '', '', '', '', '', 'subset']
            with open(path.join(root_path, 'text_annotations/a2d_missed_videos.txt'), 'r') as f:
                unused_videos = f.read().splitlines()
            subsets = {'train': 0, 'test': 1}
            # filter unused videos and videos which do not belong to our train/test subset:
            used_videos = a2d_data_info[
                ~a2d_data_info.vid.isin(unused_videos) & (a2d_data_info.subset == subsets[subset])]
            used_videos_ids = list(used_videos['vid'])
            text_annotations = pandas.read_csv(path.join(root_path, 'text_annotations/a2d_annotation.txt'))
            assert len(text_annotations) == 6655, 'error: a2d_annotations.txt is missing one or more samples.'
            # filter the text annotations based on the used videos:
            used_text_annotations = text_annotations[text_annotations.video_id.isin(used_videos_ids)]
            # remove a single dataset annotation mistake in video: T6bNPuKV-wY
            used_text_annotations = used_text_annotations[used_text_annotations['instance_id'] != '1 (copy)']
            # convert data-frame to list of tuples:
            used_text_annotations = list(used_text_annotations.to_records(index=False))
            # different from the same function in create_gt_in_coco_format.py
            # the aim of the code below is to get the frame_id for each h5 text annotations
            text_annotations_by_frame = []
            mask_annotations_dir = path.join(root_path, 'text_annotations/a2d_annotation_with_instances')
            #print(used_text_annotations)
            for video_id, instance_id, query in tqdm(used_text_annotations):
                frame_annot_paths = sorted(glob(path.join(mask_annotations_dir, video_id, '*.h5')))
                short_text_query = A2DSentencesDataset.generate_short_text_query(query)#Changes made by Dedeep.v.: Added a function to generate short text query
                for p in frame_annot_paths:
                    #print("in p for",p)
                    f = h5py.File(p)
                    instances = list(f['instance'])
                    #print("Instances",instances)
                    #print(instance_id in instances)
                    if (np.int64(instance_id) in instances):#Changes made by Dedeep.v.: converted instance_id to int64
                        frame_idx = int(p.split(os.sep)[-1].split('.')[0])
                        #print("Query",query)
                        short_text_query = short_text_query.lower()
                        #print("Short Text Query",short_text_query)
                        text_annotations_by_frame.append((query, short_text_query, video_id, frame_idx, instance_id))
            with open(saved_annotations_file_path, 'w') as f:
                json.dump(text_annotations_by_frame, f)
            #print("text_annotations_by_frame",text_annotations_by_frame)    
            return text_annotations_by_frame
        if distributed:
            dist.barrier()
            with open(saved_annotations_file_path, 'r') as f:
                text_annotations_by_frame = [tuple(a) for a in json.load(f)]
        return text_annotations_by_frame

    def __getitem__(self, idx):
        text_query, short_text_query, video_id, frame_idx, instance_id = self.text_annotations[idx]

        text_query = " ".join(text_query.lower().split())  # clean up the text query
        #print("Text Query",text_query)
        short_text_query = " ".join(short_text_query.lower().split())
        #print("Short Text Query",short_text_query)

        # read the source window frames:
        video_frames, _, _ = read_video(path.join(self.videos_dir, f'{video_id}.mp4'), pts_unit='sec')  # (T, H, W, C)
        #print("Video_frame_list: ", video_frames)
        # get a window of window_size frames with frame frame_idx in the middle.
        # note that the original a2d dataset is 1 indexed, so we have to subtract 1 from frame_idx
        start_idx, end_idx = frame_idx - 1 - self.window_size // 2, frame_idx - 1 + (self.window_size + 1) // 2

        # extract the window source frames:
        source_frames = []
        for i in range(start_idx, end_idx):
            i = min(max(i, 0), len(video_frames)-1)  # pad out of range indices with edge frames
            source_frames.append(F.to_pil_image(video_frames[i].permute(2, 0, 1))) # [Window, C, H, W]

        # read the instance mask:
        frame_annot_path = path.join(self.mask_annotations_dir, video_id, f'{frame_idx:05d}.h5')
        f = h5py.File(frame_annot_path, 'r')
        instances = list(f['instance'])
        # print("Instances",instances)
        # print("Instance element type",type(instances[0]))
        # print("Instance_id",instance_id)
        # print("type of instance_id",type(instance_id))
        #Changes made by Dedeep.v.: converted instance_id to int64
        instance_idx = instances.index(np.int64(instance_id))  # existence was already validated during init
        # print("Instance_idx",instance_idx)

        instance_masks = np.array(f['reMask'], dtype=np.uint8)
        if len(instances) == 1:
            instance_masks = instance_masks[np.newaxis, ...]
        # (num_instance, W, H) -> (num_instance, H, W)
        instance_masks = torch.tensor(instance_masks).transpose(1, 2)
        #print(len(instances), " ",instance_idx, " ",instance_masks.shape)
        mask_rles = [encode(mask) for mask in instance_masks.numpy()]
        mask_areas = area(mask_rles).astype(float)#Changes made by Dedeep.v.: edited astype(np.float) to astype(float) 
        f.close()

        # create the target dict for the center frame:
        target = {'masks': instance_masks,
                  'orig_size': instance_masks.shape[-2:],  # original frame shape without any augmentations
                  # size with augmentations, will be changed inside transforms if necessary
                  'size': instance_masks.shape[-2:],
                  'referred_instance_idx': torch.tensor(instance_idx),  # idx in 'masks' of the text referred instance
                  'area': torch.tensor(mask_areas),
                  'iscrowd': torch.zeros(len(instance_masks)),  # for compatibility with DETR COCO transforms
                  'image_id': get_image_id(video_id, frame_idx, instance_id)}

        # create dummy targets for adjacent frames:
        targets = self.window_size * [None]
        center_frame_idx = self.window_size // 2
        targets[center_frame_idx] = target
        source_frames, targets, text_query, short_text_query = self.transforms(source_frames, targets, text_query, short_text_query)
        return source_frames, targets, text_query, short_text_query

    def __len__(self):
        return len(self.text_annotations)


class A2dSentencesTransforms:
    def __init__(self, subset_type, horizontal_flip_augmentations, resize_and_crop_augmentations,
                 train_short_size, train_max_size, eval_short_size, eval_max_size, **kwargs):
        self.h_flip_augmentation = subset_type == 'train' and horizontal_flip_augmentations
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        scales = [train_short_size]  # no more scales for now due to GPU memory constraints. might be changed later
        transforms = []
        if resize_and_crop_augmentations:
            if subset_type == 'train':
                transforms.append(T.RandomResize(scales, max_size=train_max_size))
            elif subset_type == 'test':
                transforms.append(T.RandomResize([eval_short_size], max_size=eval_max_size)),
        transforms.extend([T.ToTensor(), normalize])
        self.size_transforms = T.Compose(transforms)

    def __call__(self, source_frames, targets, text_query, short_text_query):
        if self.h_flip_augmentation and torch.rand(1) > 0.5:
            source_frames = [F.hflip(f) for f in source_frames]
            targets[len(targets) // 2]['masks'] = F.hflip(targets[len(targets) // 2]['masks'])
            # Note - is it possible for both 'right' and 'left' to appear together in the same query. hence this fix:
            text_query = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
            short_text_query = short_text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
        source_frames, targets = list(zip(*[self.size_transforms(f, t) for f, t in zip(source_frames, targets)]))
        source_frames = torch.stack(source_frames)  # [T, 3, H, W]
        return source_frames, targets, text_query, short_text_query


class Collator:
    def __call__(self, batch):
        samples, targets, text_queries, short_text_queries = list(zip(*batch))
        # len(samples) = B; Bx [T,C,H,W] -> [T,B,C,H,W]
        # padding mask  [T,B,H,W]
        samples = nested_tensor_from_videos_list(samples)  # [T, B, C, H, W] & [T, B, H, W]
        # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
        targets = list(zip(*targets))
        batch_dict = {
            'samples': samples,
            'targets': targets,
            'text_queries': text_queries,
            'short_text_queries': short_text_queries
        }
        return batch_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MTTR training and evaluation')
    parser.add_argument('--config_path', '-c', required=True,
                        help='path to configuration file')
    parser.add_argument('--running_mode', '-rm', choices=['train', 'eval'], required=True,
                        help="mode to run, either 'train' or 'eval'")
    parser.add_argument('--window_size', '-ws', type=int,
                        help='window length to use during training/evaluation.'
                             'note - in Refer-YouTube-VOS this parameter is used only during training, as'
                             ' during evaluation full-length videos (all annotated frames) are used.')
    parser.add_argument('--batch_size', '-bs', type=int, required=True,
                        help='training batch size per device')
    parser.add_argument('--eval_batch_size', '-ebs', type=int,
                        help='evaluation batch size per device. '
                             'if not provided training batch size will be used instead.')
    parser.add_argument('--checkpoint_path', '-ckpt', type=str,
                        help='path of checkpoint file to load for evaluation purposes')
    gpu_params_group = parser.add_mutually_exclusive_group(required=True)
    gpu_params_group.add_argument('--num_gpus', '-ng', type=int, default=argparse.SUPPRESS,
                                  help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    gpu_params_group.add_argument('--gpu_ids', '-gids', type=int, nargs='+', default=argparse.SUPPRESS,
                                  help='ids of GPUs to run on. mutually exclusive with \'num_gpus\'')
    gpu_params_group.add_argument('--cpu', '-cpu', action='store_true', default=argparse.SUPPRESS,
                                  help='run on CPU. Not recommended, but could be helpful for debugging if no GPU is'
                                       'available.')
    args = parser.parse_args()

    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    if hasattr(args, 'num_gpus'):
        args.num_devices = max(min(args.num_gpus, torch.cuda.device_count()), 1)
        args.device_ids = list(range(args.num_gpus))
    elif hasattr(args, 'gpu_ids'):
        for gpu_id in args.gpu_ids:
            assert 0 <= gpu_id <= torch.cuda.device_count() - 1, \
                f'error: gpu ids must be between 0 and {torch.cuda.device_count() - 1}'
        args.num_devices = len(args.gpu_ids)
        args.device_ids = args.gpu_ids
    else:  # cpu
        args.device_ids = ['cpu']
        args.num_devices = 1

    # print(args)  # delete!
    with open(args.config_path) as f:
        config = ruamel.yaml.safe_load(f)
    config = {k: v['value'] for k, v in config.items()}
    config = {**config, **vars(args)}
    config = argparse.Namespace(**config)
    dataset_train = A2DSentencesDataset(subset_type='train', **vars(config))
    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, sampler=None,
               collate_fn=dataset_train.collator, num_workers=config.num_workers,
               pin_memory=True, shuffle=True)
    for batch_dict in tqdm(dataloader_train):
        samples = batch_dict['samples']
        targets = batch_dict['targets']
        print('--------------------------------------------------------------------')
        text_queries = batch_dict['text_queries']
        short_text_queries = batch_dict['short_text_queries']
        print(text_queries)
        print(short_text_queries)
        print('--------------------------------------------------------------------')
    print('end')
