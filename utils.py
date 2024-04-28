from PIL import Image
import numpy as np
import cv2

def getScreenFromEnv(env):
    env.render()
    screen = env.render("rgb_array")
    return Image.fromarray(screen)

def cropImage(img, targetHeight):
    resizedImage = cv2.resize(img, (110, 84))

    target_width = int(targetHeight * resizedImage.shape[1] / resizedImage.shape[0])
    cropLeft = (target_width - 84) // 2
    cropRight = cropLeft + 84
    croppedImage = resizedImage[:, cropLeft:cropRight]
    return croppedImage

def standardPreprocess(grayscaleImage):
    downSampled = cv2.resize(grayscaleImage, (110, 84))
    return np.array(cropImage(downSampled, 84))