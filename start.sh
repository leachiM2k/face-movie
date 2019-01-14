#!/bin/bash
python main.py
ffmpeg -framerate 10 -pattern_type glob -i 'output/face_*.png' -c:v libx264 -pix_fmt yuv420p out.mp4

echo "Facemovie is ready! (out.mp4)"
