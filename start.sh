#!/bin/bash
echo "++++++++++++++ Executing Face Recognition and Face Alignment"

rm -rf playground/output/c_*
python main.py

echo "++++++++++++++ Combining to movie"

rm -f payload/out_morphed.mp4
ffmpeg -framerate 10 -pattern_type glob -i 'payload/output/c_*.j*' -vf "minterpolate=fps=30:mi_mode=blend,pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p payload/out_morphed.mp4

echo "++++++++++++++ Facemovie is ready! (out_morphed.mp4)"
