import os, glob
from pathlib import Path
import subprocess
def video_frame_extractor(root_dir):
    class_folders=[name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))] # Choose folder only
    class_folders=[class_folder for class_folder in class_folders if class_folder.split("_")[-1]!='frames'] # remove extracted frame folders
    for class_folder in class_folders:
        a_path=os.path.join(root_dir,class_folder)
        video_files=os.listdir(a_path)
        if len(video_files)>0:
            class_save_dir=a_path+'_frames'
            Path(class_save_dir).mkdir(parents=True, exist_ok=True)
            for video_file in video_files:
                video_save_path = os.path.join(class_save_dir,video_file.split(".")[0])
                Path(video_save_path).mkdir(parents=True, exist_ok=True)
                video_path = os.path.join(a_path,video_file)
                subprocess.call(['ffmpeg', '-i', video_path, '-vf', 'fps=30', os.path.join(video_save_path,'%04d.png')])

if __name__ == '__main__':
    root_dir="/nfs/users/sohel/cricket_shot_dataset"
    video_frame_extractor(root_dir)
