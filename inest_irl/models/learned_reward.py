import numpy as np
from tqdm import tqdm

from inest_irl.maniskill3.stack_pyramid import MAX_SUBGOAL


DEFAULT_C_VALUE = 0.25   # additional reward for reaching any subgoal
DEFAULT_DISTANCE_THRESHOLDS = [0.5, 0.5, 0.5, 0.5]  # distance threshold for considering a subgoal reached (in embedding space)
DEFAULT_PATIENCE_THRESHOLD = 2


def compute_goal_embedding(
  model,
  train_loader,
  subgoal_frames,
  device,
  c_value=DEFAULT_C_VALUE,
  distance_thresholds=DEFAULT_DISTANCE_THRESHOLDS,
  patience_threshold=DEFAULT_PATIENCE_THRESHOLD,
):
  """Compute the mean goal embedding from the last frames of trajectories"""
  init_embs, goal_embs, subgoal_embs_list = [], [], []

  # get init, goal (final), and subgoals embeddings for each trajectory in the training set
  for class_name, class_loader in train_loader.items():
    for batch in tqdm(iter(class_loader), leave=True, desc=f"Embedding {class_name}"):
      out = model.infer(batch["frames"].to(device))   # batch_size=1 since hardcoded in downstream dataloader
      emb = out.numpy().embs  # shape: (seq_len, embedding_dim)
      
      init_embs.append(emb[0, :])   # first frame embedding
      goal_embs.append(emb[-1, :])  # last frame embedding

      if subgoal_frames is not None:
        traj_id = batch["video_name"][0].split('/')[-1]  # video name should be in format .../../video_id
        
        # skip if trajectory ID not in subgoal_frames (data mismatch)
        if traj_id not in subgoal_frames:
          print(f"Warning: Trajectory ID {traj_id} not found in subgoal frames data - skipping subgoal embedding for this trajectory")
          continue
        
        subgoal_idxs = subgoal_frames[traj_id]

        # if empty list, add empty lists inside with the length of the number of subgoals
        if len(subgoal_embs_list) == 0:
          for _ in range(MAX_SUBGOAL):
            subgoal_embs_list.append([])

        # add subgoal embeddings to the corresponding subgoal index list
        for i, idx in enumerate(subgoal_idxs):
          if idx >= emb.shape[0]:  # sanity check for subgoal index out of bounds
            print(f"Warning: Subgoal index {idx} for trajectory {traj_id} is out of bounds (trajectory length {emb.shape[0]}) - skipping this subgoal")
            continue
          subgoal_embs_list[i].append(emb[idx, :])  # subgoal frame embedding
  
  # compute mean goal embedding and distance scale
  goal_emb = np.mean(np.stack(goal_embs, axis=0), axis=0, keepdims=True)
  dist_to_goal = np.linalg.norm(np.stack(init_embs, axis=0) - goal_emb, axis=1).mean()
  dist_scale = 1.0 / (dist_to_goal + 1e-8)

  # compute mean subgoal embeddings if subgoal frames are provided
  if subgoal_frames is not None:
    subgoal_embs = []
    for traj_subgoal_embs in subgoal_embs_list:
      subgoal_embs.append(np.mean(np.stack(traj_subgoal_embs, axis=0), axis=0, keepdims=True))
  else:
    subgoal_embs = None
    print("WARNING: No subgoal embeddings computed")
  
  # add subgoal info for pickling, used by wrapper in rl training
  subgoal_info = {
    "c_value": c_value,
    "distance_thresholds": distance_thresholds,
    "patience_threshold": patience_threshold,
  }
  
  return goal_emb, subgoal_embs, dist_scale, subgoal_info


# TODO: move reward computation functions here as well