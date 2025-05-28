import json
import h5py
import torch

from torch.utils.data import Dataset
from torchvision.io import read_video
from os import path
from glob import glob
from tqdm import tqdm
from pycocotools.mask import encode, area
from misc import nested_tensor_from_videos_list
from nltk.tokenize import word_tokenize
from nltk import pos_tag

class ReferDavisDataset(Dataset):
    """
    A Torch dataset for Refer-DAVIS.
    """
    def __init__(self, subset_type: str = 'train', dataset_path: str = './refer_davis', window_size=8, **kwargs):
        super(ReferDavisDataset, self).__init__()
        assert subset_type in ['train', 'test'], 'error, unsupported dataset subset type. supported: train, test'
        self.subset_type = subset_type
        self.mask_annotations_dir = path.join(dataset_path, 'davis_text_annotations')
        self.videos_dir = path.join(dataset_path, 'DAVIS-2017-Unsupervised-trainval-480p')
        self.text_annotations = self.get_text_annotations(dataset_path, subset_type)
        self.window_size = window_size
        self.transforms = None  # Add your transforms here if needed

    @staticmethod
    def get_text_annotations(root_path, subset):
        """
        Load text annotations for the given subset, dynamically handling all relevant files.
        """
        annotations_dir = path.join(root_path, 'davis_text_annotations')
        annotations = []

        for file_name in glob(path.join(annotations_dir, f'*{subset}*.txt')):
            with open(file_name, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 2)
                    if len(parts) == 3:
                        video_id, frame_idx, query = parts
                        annotations.append((video_id, int(frame_idx), query.strip('"')))

        if not annotations:
            raise ValueError(f"No valid annotations found in {annotations_dir} for subset {subset}.")

        return annotations

    @staticmethod
    def generate_short_text_query(query):
        words = word_tokenize(query) 
        pos_tags = pos_tag(words)  
        verb_index = next((i for i, (_, tag) in enumerate(pos_tags) if tag.startswith('VB')), len(pos_tags))
        short_text = " ".join(words[:verb_index])
        return short_text

    def __getitem__(self, idx):
        video_id, frame_idx, instance_id = self.text_annotations[idx]

        # Retrieve long and short queries
        long_query = self.long_queries.get((video_id, frame_idx), "")
        short_query = self.short_queries.get((video_id, frame_idx), "")

        # Read video frames
        video_frames, _, _ = read_video(path.join(self.videos_dir, f'{video_id}.mp4'), pts_unit='sec')

        # Get a window of frames
        start_idx, end_idx = frame_idx - self.window_size // 2, frame_idx + self.window_size // 2
        source_frames = [video_frames[min(max(i, 0), len(video_frames)-1)] for i in range(start_idx, end_idx)]

        # Read instance mask
        frame_annot_path = path.join(self.mask_annotations_dir, video_id, f'{frame_idx:05d}.h5')
        with h5py.File(frame_annot_path, 'r') as f:
            instances = list(f['instance'])
            instance_idx = instances.index(instance_id)
            instance_masks = torch.tensor(f['reMask']).transpose(1, 2)

        # Create target dict
        target = {
            'masks': instance_masks,
            'orig_size': instance_masks.shape[-2:],
            'size': instance_masks.shape[-2:],
            'referred_instance_idx': torch.tensor(instance_idx),
        }

        return source_frames, target, long_query, short_query

    def __len__(self):
        return len(self.text_annotations)

    @staticmethod
    def collator(batch):
        """
        Custom collator function to handle batching for ReferDavisDataset.
        """
        source_frames, targets, long_queries, short_queries = zip(*batch)

        # Stack frames and targets
        batched_frames = nested_tensor_from_videos_list(source_frames)
        batched_targets = [{
            'masks': torch.stack([t['masks'] for t in targets]),
            'orig_size': torch.stack([torch.tensor(t['orig_size']) for t in targets]),
            'size': torch.stack([torch.tensor(t['size']) for t in targets]),
            'referred_instance_idx': torch.stack([t['referred_instance_idx'] for t in targets]),
        }]

        # Include scribbles if available
        scribbles = [t.get('scribbles', None) for t in targets]
        if any(scribbles):
            batched_targets[0]['scribbles'] = scribbles

        return batched_frames, batched_targets, list(long_queries), list(short_queries)
