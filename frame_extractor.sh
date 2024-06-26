#!/bin/bash

# Define video directory (replace with your actual video directory path)
video_dir="/home/ubuntu/cricket-shot-classification-training/shot-classification-video/cover_drive"

for file in "$video_dir"/*; do
  # Check if the file is a video using its extension
  if [[ $file == *.mp4 || $file == *.avi || $file == *.mkv ]]; then 
      # Extract filename without extension
      filename="${file%.*}"

      # Create a folder for the video frames (if it doesn't exist)
      mkdir -p "$filename"

      # Convert video to frames and store in the folder
      ffmpeg -i "$file" "$filename"/%d.jpg

      # Optional: Display processing information
      echo "Converted $file to frames."
  fi
done

echo "All videos processed in $video_dir"
