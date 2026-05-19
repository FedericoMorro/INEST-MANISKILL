"""
Merge multiple videos into a single video with a grid layout using PIL and ffmpeg.
"""

import argparse
import os
import subprocess
import tempfile
from PIL import Image
import numpy as np
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except Exception:
    import imageio


def get_video_frames(video_path, temp_dir, video_id=0):
    """Extract frames from video using ffmpeg."""
    abs_path = os.path.abspath(video_path)
    if not os.path.exists(abs_path):
        raise ValueError(f"Video file not found: {abs_path}")
    
    # Create subdirectory for this video's frames using video_id for uniqueness
    frames_dir = os.path.join(temp_dir, f"video_{video_id}")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Extract frames using ffmpeg
    cmd = [
        'ffmpeg', '-loglevel', 'error', '-i', abs_path, '-q:v', '2',
        os.path.join(frames_dir, 'frame_%06d.png')
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise ValueError(f"ffmpeg failed to extract frames: {result.stderr}")
    
    # Load frames with PIL
    frames = []
    frame_idx = 1  # ffmpeg frame numbering starts at 1, not 0
    while True:
        frame_path = os.path.join(frames_dir, f'frame_{frame_idx:06d}.png')
        if not os.path.exists(frame_path):
            break
        frames.append(Image.open(frame_path).convert('RGB'))
        frame_idx += 1
    
    if not frames:
        raise ValueError(f"Could not extract frames from {abs_path}")
    
    return frames


def create_grid_frames(batch_videos, grid_size, frame_width, frame_height, temp_dir, batch_start_idx=0):
    """Create grid frames from a batch of videos."""
    grid_rows, grid_cols = grid_size
    merged_width = frame_width * grid_cols
    merged_height = frame_height * grid_rows
    
    # Extract frames from all videos in batch
    all_frames = []
    for video_idx, video_path in tqdm(enumerate(batch_videos), desc="Extracting frames", total=grid_rows * grid_cols, leave=False):
        frames = get_video_frames(video_path, temp_dir, video_id=batch_start_idx + video_idx)
        all_frames.append(frames)
    
    # Get max number of frames (pad shorter videos with last frame)
    max_frames = max(len(frames) for frames in all_frames) if all_frames else 0
    
    # Pad shorter videos with their last frame
    for frames_list in tqdm(all_frames, desc="Padding frames", leave=False):
        if len(frames_list) > 0:
            last_frame = frames_list[-1]
            while len(frames_list) < max_frames:
                frames_list.append(last_frame)
    
    # Create merged frames
    merged_frames = []
    for frame_idx in tqdm(range(max_frames), desc="Creating merged frames", leave=False):
        merged_img = Image.new('RGB', (merged_width, merged_height), color='black')
        
        for video_idx, frames_list in enumerate(all_frames):
            row = video_idx // grid_cols
            col = video_idx % grid_cols
            
            x = col * frame_width
            y = row * frame_height
            merged_img.paste(frames_list[frame_idx], (x, y))
        
        merged_frames.append(merged_img)
    
    return merged_frames, (merged_width, merged_height)


def merge_videos_batch(video_paths, output_path, grid_size=(4, 4), fps=10):
    """Merge multiple videos into a single video with batched grid layout.
    
    Videos are organized into batches (e.g., 16 for 4x4 grid) and displayed sequentially.
    Each batch is shown as a grid, then the next batch, etc. All batches are concatenated
    into a single output video.
    """
    if not video_paths:
        raise ValueError("No video paths provided")
    
    num_videos = len(video_paths)
    grid_rows, grid_cols = grid_size
    batch_size = grid_rows * grid_cols
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Get frame dimensions from first video
        first_frames = get_video_frames(video_paths[0], temp_dir, video_id=0)
        if not first_frames:
            raise ValueError(f"Could not read frames from {video_paths[0]}")
        frame_width, frame_height = first_frames[0].size

        # Process videos in batches
        num_batches = (num_videos + batch_size - 1) // batch_size
        
        all_batch_videos = []
        
        print(f"Script will process {num_batches} batches corresponding to videos (batch_id: video idx_start-idx_end):")
        for batch_idx in range(num_batches):
            print(f"{batch_idx}: {batch_idx * batch_size + 1}-{min((batch_idx + 1) * batch_size, num_videos)}    ", end="")
        print()

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_videos)
            batch_videos = video_paths[start_idx:end_idx]
            
            # Create grid frames for this batch
            merged_frames, (merged_width, merged_height) = create_grid_frames(
                batch_videos, grid_size, frame_width, frame_height, temp_dir, batch_start_idx=start_idx
            )
            
            # Save batch frames to temporary files
            batch_temp_dir = os.path.join(temp_dir, f"merged_batch_{batch_idx}")
            os.makedirs(batch_temp_dir, exist_ok=True)
            
            for frame_idx, frame in tqdm(enumerate(merged_frames), desc="Saving batch frames", total=len(merged_frames), leave=False):
                frame.save(os.path.join(batch_temp_dir, f"frame_{frame_idx:06d}.png"))
            
            # Create video from frames using ffmpeg
            batch_video_path = os.path.join(temp_dir, f"batch_{batch_idx}.mp4")
            ffmpeg_cmd = [
                'ffmpeg', '-loglevel', 'error', '-y', '-framerate', str(fps),
                '-i', os.path.join(batch_temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-q:v', '2',
                batch_video_path
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError(f"ffmpeg failed to create batch video: {result.stderr}")
            all_batch_videos.append(batch_video_path)
        
        # Concatenate all batch videos
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            for batch_video in all_batch_videos:
                f.write(f"file '{batch_video}'\n")
        
        cmd = [
            'ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file, '-c', 'copy', output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"ffmpeg failed to concatenate videos: {result.stderr}")
    
    print(f"Completed! Created merged video with {num_batches} batch(es) in {output_path}")
    
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Merge multiple videos into a single video with batched grid layout.")
    argparser.add_argument("--video_dir", type=str, required=True, help="Directory containing input videos.")
    argparser.add_argument("--output_dir", type=str, default=None, help="Directory to save the merged video.")
    argparser.add_argument("--output_name", type=str, default="_merged", help="Filename for the merged video.")
    argparser.add_argument("--grid_dims", type=int, nargs=2, default=(4, 4), help="Grid dimensions (rows cols) for merging videos.")
    argparser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
    args = argparser.parse_args()
    
    video_paths = sorted([os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(('.mp4', '.avi'))])
    
    if not video_paths:
        print(f"No videos found in {args.video_dir}")
    else:
        print(f"Found {len(video_paths)} videos")
        if args.output_dir is None:
            output_path = os.path.join(args.video_dir, f"{args.output_name}.mp4")
        else:
            output_path = os.path.join(args.output_dir, f"{args.output_name}.mp4")
        merge_videos_batch(video_paths, output_path, grid_size=tuple(args.grid_dims), fps=args.fps)