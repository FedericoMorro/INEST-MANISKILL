from absl import logging
import os
import torch
import torch.nn as nn

from xirl.dataset import VideoDataset, SequenceType
import xirl.factory as factory
from xirl.models import SelfSupervisedModel, SelfSupervisedOutput
import xirl.video_samplers as video_samplers



class MultipleCamerasModel(SelfSupervisedModel):
    """Different separate encoders for different camera views, with a shared fusion module to combine the embeddings."""
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # store sorted camera names for consistent indexing
        self.sorted_camera_names = sorted(config.camera_names)
        
        # create separate encoders for each camera, keyed by name
        self.encoders = nn.ModuleDict({})
        for camera_name in self.sorted_camera_names:
            self.encoders[camera_name] = factory.model_from_config(config)
        
        # simple fusion module that concatenates the embeddings and passes through an MLP
        self.fusion_module = nn.Sequential(
            nn.Linear(len(self.sorted_camera_names) * config.model.embedding_size, config.model.embedding_size),
            nn.ReLU(),
            nn.Linear(config.model.embedding_size, config.model.embedding_size),
        )
        
    def forward(self, inputs):
        # inputs is a tensor with shape (batch_size, num_cameras, num_frames, channels, height, width)
        batch_size, num_cameras, num_frames, channels, height, width = inputs.shape
        
        embeddings = []
        for camera_idx in range(num_cameras):
            # Extract frames for this camera: shape (batch_size, num_frames, channels, height, width)
            camera_input = inputs[:, camera_idx, :, :, :, :]
            encoder_key = self.sorted_camera_names[camera_idx]
            camera_emb = self.encoders[encoder_key](camera_input).embs  # (batch_size, num_frames, embedding_dim)
            embeddings.append(camera_emb)
        
        # concatenate embeddings from different cameras: (batch_size, num_frames, num_cameras * embedding_dim)
        concat_emb = torch.cat(embeddings, dim=-1)
        fused_emb = self.fusion_module(concat_emb)  # (batch_size, num_frames, embedding_dim)
        
        # reshape frames from (batch_size, num_cameras, num_frames, C, H, W) to (batch_size, num_frames, num_cameras*C, H, W)
        frames_reshaped = inputs.permute(0, 2, 1, 3, 4, 5).reshape(batch_size, num_frames, -1, height, width)
        
        return SelfSupervisedOutput(frames=frames_reshaped, feats=concat_emb, embs=fused_emb)
    

class MultipleCamerasMatchedBatchSampler(video_samplers.VideoBatchSampler):
    """Batch sampler for matched trajectory indices across different camera views.
    
    Returns batches where all cameras are represented for each matched video index.
    Cameras are indexed by their sorted order (0, 1, ..., num_cameras-1).
    e.g. batch_size=3, 2 cameras -> [(0,5), (1,5), (0,6), (1,6), (0,7), (1,7)]
    """
    
    def __init__(self, num_cameras, data_len, **kwargs):
        super().__init__(**kwargs)
        self.num_cameras = num_cameras
        self.data_len = data_len

    def __len__(self):
        return self.data_len
    
    def _generate_indices(self):
        # get video lists for each camera (assumed to be in consistent order)
        camera_sequences = {}
        camera_paths = sorted(self._dir_tree.keys())
        
        if len(camera_paths) != self.num_cameras:
            raise ValueError(
                f"Number of camera paths ({len(camera_paths)}) doesn't match "
                f"num_cameras ({self.num_cameras})"
            )
        
        # iterate through cameras by index using sorted paths
        for camera_idx, camera_path in enumerate(camera_paths):
            video_list = self._dir_tree[camera_path]
            len_v = len(video_list)
            seq = list(range(len_v))
            
            # shuffle only once (based on first camera) to ensure matching
            if not self._sequential and camera_idx == 0:
                perm = torch.randperm(len(seq))
                seq = [seq[i] for i in perm]
            
            camera_sequences[camera_idx] = seq
        
        num_videos = len(camera_sequences[0])
        
        # create batches with (camera_idx, video_idx) tuples
        batch_indices = []
        for batch_start in range(0, num_videos - self._batch_size + 1, self._batch_size):
            batch = []
            for video_offset in range(self._batch_size):
                video_idx = camera_sequences[0][batch_start + video_offset]
                for camera_idx in range(self.num_cameras):
                    batch.append((camera_idx, video_idx))
            batch_indices.append(batch)
        
        return batch_indices
        
    
def collate_fn_multiple_cameras(batch, num_cameras):
    """Collate function for multi-camera batches.
    
    Receives fetched items from dataset in interleaved order:
    [cam0_item0, cam1_item0, ..., camN_item0, cam0_item1, cam1_item1, ...]
    
    Returns a dict with:
    - "frames": stacked frames tensor with shape (batch_size, num_cameras, ...)
    - "frame_idxs": frame indices with shape (batch_size,) [same across cameras]
    - "video_len": video lengths with shape (batch_size,) [same across cameras]
    - "video_names": list of video names [same across cameras]
    """
    # group items by camera index
    camera_items = {i: [] for i in range(num_cameras)}
    for item_idx, item in enumerate(batch):
        camera_idx = item_idx % num_cameras
        camera_items[camera_idx].append(item)
    
    def _collate_camera_items(items):
        """Collate a list of items from a single camera."""
        result = {}
        
        # stack tensor-based keys
        for key in [SequenceType.FRAMES, SequenceType.FRAME_IDXS, SequenceType.VIDEO_LEN]:
            result[str(key)] = torch.stack([item[key] for item in items])
        
        # keep VIDEO_NAME as list of strings
        result[str(SequenceType.VIDEO_NAME)] = [item[SequenceType.VIDEO_NAME] for item in items]
        
        return result
    
    # collate all cameras
    all_camera_frames = []
    for camera_idx in range(num_cameras):
        camera_data = _collate_camera_items(camera_items[camera_idx])
        all_camera_frames.append(camera_data[str(SequenceType.FRAMES)])
    
    # stack frames: (batch_size, num_cameras, ...) instead of (num_cameras, batch_size, ...)
    output = {
        str(SequenceType.FRAMES): torch.stack(all_camera_frames, dim=1),
        str(SequenceType.FRAME_IDXS): _collate_camera_items(camera_items[0])[str(SequenceType.FRAME_IDXS)],
        str(SequenceType.VIDEO_LEN): _collate_camera_items(camera_items[0])[str(SequenceType.VIDEO_LEN)],
        str(SequenceType.VIDEO_NAME): _collate_camera_items(camera_items[0])[str(SequenceType.VIDEO_NAME)],
    }
    
    return output
    
    
def get_dataloaders_mc(config, downstream=False, debug=False):
    """Get multi-camera dataloaders.
    
    For pretraining: Returns {"train": DataLoader, "valid": DataLoader}
    For downstream evaluation: Returns {"train": {"multicamera": DataLoader}, "valid": {"multicamera": DataLoader}}
    This structure matches the evaluator's expectations: for split, downstream_loader in loaders.items()
    then for action_name, valid_loader in downstream_loader.items()
    """
    
    def _loader(split):
        # for pretraining, use frame sampler (downstream=False), for downstream, request full trajectories (downstream=True)
        dataset = factory.dataset_from_config(config, downstream, split, debug, multi_camera=True)
        
        num_cameras = len(config.camera_names)
        
        # dataset.py: VideoDataset.total_vids (220)
        #   override value by dividing it by the number of cameras, since each video is duplicated across cameras in the dataset
        #   and the count is performed directly on the dir_tree which contains all cameras' videos
        #   then enforce len also in loader
        dataset.total_vids = dataset.total_vids // num_cameras
        
        kwargs = {
            "dir_tree": dataset.dir_tree,
            "batch_size": config.data.batch_size,
            "sequential": debug,
        }
        if downstream:
            kwargs["batch_size"] = 1
        
        batch_sampler = MultipleCamerasMatchedBatchSampler(num_cameras, len(dataset), **kwargs)
        
        # create collate function with num_cameras bound
        collate_fn = lambda batch: collate_fn_multiple_cameras(batch, num_cameras)
        
        loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            num_workers=0,  # avoid multiprocessing temp cleanup errors on NFS
            pin_memory=torch.cuda.is_available() and not debug,
        )
        
        # for downstream evaluation, wrap loader in dict with action name
        if downstream:
            return {"multicamera": loader}
        else:
            return loader
    

    return {
        "train": _loader("train"),
        "valid": _loader("valid"),
    }
    

def get_model_mc(config):
    kwargs = {
        "num_ctx_frames": config.frame_sampler.num_context_frames,
        "normalize_embeddings": config.model.normalize_embeddings,
        "learnable_temp": config.model.learnable_temp,
    }
    return MultipleCamerasModel(config, **kwargs)
    

def get_factories_mc(config, debug=False):
    """Feed config to factories and return objects for multi-camera setting."""
    pretrain_loaders = get_dataloaders_mc(config, debug=debug)
    downstream_loaders = get_dataloaders_mc(config, downstream=True, debug=debug)
    model = get_model_mc(config)
    
    return pretrain_loaders, downstream_loaders, model