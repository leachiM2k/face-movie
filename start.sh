#!/bin/bash
echo "++++++++++++++ Executing Face Recognition and Face Alignment"

rm -rf playground/output/c_*
python main.py

echo "++++++++++++++ Combining to movie"

rm -f payload/out_morphed.mp4
ffmpeg -f concat -safe 0 -i <( for f in payload/output/*.jp*; do echo "file '$(pwd)/$f'"; done ) -framerate 10 -vf "minterpolate=fps=30:mi_mode=blend,pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p payload/out_morphed.mp4

echo "++++++++++++++ Facemovie is ready! (out_morphed.mp4)"
