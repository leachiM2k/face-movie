# Face-Movie

FaceMovie creates movies from a bunch of jpegs.

Steps in detail:
- detects faces on images from input directory
- centers images using faces as gravitation point
- combines all images to a single mp4

Tools and libraries used:
- Python
- OpenCV
- dlib
- ffmpeg

Based on [@leoneckert](https://github.com/leoneckert/facemash-workshop) Facemesh workshop.

The landmarks dat file `shape_predictor_68_face_landmarks.dat` (approx. 95 MB) is needed for face detection.
Due to unclear copyright you should bother Google and download it from somewhere.

