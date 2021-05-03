#!/bin/bash
echo "++++++++++++++ Executing Face Recognition and Face Alignment"

rm -rf playground/output/c_*
python main.py

echo "++++++++++++++ Calculating fade images"

rm -rf morph/
mkdir morph
convert payload/output/c_* -delay 1 -morph 1 morph/%05d.morph.jpg

echo "++++++++++++++ Combining to movie"

rm -f payload/out_morphed.mp4
ffmpeg -framerate 10 -pattern_type glob -i 'morph/*.jpg' -c:v libx264 -pix_fmt yuv420p payload/out_morphed.mp4

echo "++++++++++++++ Facemovie is ready! (out_morphed.mp4)"
