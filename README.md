# Face-Movie

Face-Movie aligns faces and creates a movie from a bunch of jpegs
like Google Picasa did it back in 2011

__Why?__

Since August 2016, I automatically take a selfie of myself every day at 9am.
Now, 750 pictures later I wanted to make a very quick slideshow out of it
to see how I have changed in these 5 years. 

Unfortunately, I noticed that in each picture my head is in a different place
with different tilt. 
I needed a tool that would align all the pictures so that the head,
or at least the eyes, are at the same level.

Steps in detail:
- detects faces on images from input directory
- centers images using faces as gravitation point
- combines all images to a single mp4

Tools and libraries used:
- Python
- OpenCV
- dlib
- face-recognition
- ffmpeg

## Example

You will find three images of Jeff Bezos in `./payload/input`.

These images were processed by the python app and resulted in [./payload/out_morphed.mp4](./payload/out_morphed.mp4).

## Getting started

At first, put your source JPEGs to `./payload/input`.
The centered and aligned images will be put to `./payload/output` by the application.

The resulting movie (mp4) will land in `./payload` directly.

### Docker

    docker run --rm -v "${PWD}/payload:/app/payload" leachim2k/face-movie:latest

If you want to build it yourself, execute `docker build --tag leachim2k/face-movie:latest .`
*But be aware,* on my MacBook Pro late 2020 the build takes nearly 14 minutes. 

### Straight away

    pip install -r requirements.txt
    ./start.sh
