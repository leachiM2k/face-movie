from mtcnn.mtcnn import MTCNN
import face_recognition
import cv2
import numpy as np
from pathlib import Path

# initialise the detector class.
detector = MTCNN()

def detectFace(file):
    # load image and find face locations.
    image = face_recognition.load_image_file(file)
    face_locations2 = detector.detect_faces(image)
    # [
    #   {
    #   'box': [544, 166, 218, 289],
    #   'confidence': 0.9995384216308594,
    #   'keypoints': {'left_eye': (606, 270), 'right_eye': (706, 275), 'nose': (641, 328), 'mouth_left': (603, 386), 'mouth_right': (684, 389)}
    #   }
    # ]
    print(face_locations2)
    if len(face_locations2) == 0:
        return

    '''
    Let's find and angle of the face. First calculate 
    the center of left and right eye by using eye landmarks.
    '''
    leftEyeCenter = face_locations2[0]['keypoints']['left_eye']
    rightEyeCenter = face_locations2[0]['keypoints']['right_eye']

    # draw the circle at centers and line connecting to them
    (x, y, w, h) = face_locations2[0]['box']
    (x2, y2) = (x + w, y + h)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # cv2.circle(image, leftEyeCenter, 2, (255, 0, 0), 10)
    # cv2.circle(image, rightEyeCenter, 2, (255, 0, 0), 10)
    # cv2.line(image, leftEyeCenter, rightEyeCenter, (255, 0, 0), 10)

    # find and angle of line by using slop of the line.
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # to get the face at the center of the image,
    # set desired left eye location. Right eye location
    # will be found out by using left eye location.
    # this location is in percentage.
    desiredLeftEye = (0.35, 0.35)
    # Set the croped image(face) size after rotaion.

    desiredFaceWidth = 128
    desiredFaceHeight = 128

    (desiredFaceWidth, desiredFaceHeight) = (image.shape[1], image.shape[0])

    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    # desiredDist *= desiredFaceWidth
    desiredDist *= 300
    scale = desiredDist / dist
    # scale = 1

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                  (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)

    output = cv2.warpAffine(image,
                            M,
                            (w, h),
                            borderMode=cv2.BORDER_CONSTANT,
                            flags=cv2.INTER_CUBIC,
                            )
    print('Writing cropped image: c_' + file.name)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, file.name, (10, image.shape[0] - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite('payload/output/c_' + file.name, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

for file in sorted(Path("payload/input").iterdir()):
    if file.suffix.lower() == ".jpg" or file.suffix.lower() == ".jpeg":
        detectFace(file)
