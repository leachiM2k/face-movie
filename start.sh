#!/bin/bash
python main.py
ffmpeg -framerate 10 -pattern_type glob -i 'output/face_*.png' -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p out.mp4

echo "Facemovie is ready! (out.mp4)"
