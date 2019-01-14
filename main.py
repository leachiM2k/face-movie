import facemesh as fm
import cv2

def main():

    #Instantiate the class
    fullSet = "input"
    faceReader = fm.FaceMesh("shape_predictor_68_face_landmarks.dat")
    faceReader.align(fullSet)

if __name__ == '__main__':
    main()
