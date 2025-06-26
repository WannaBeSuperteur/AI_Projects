import numpy as np
import cv2


def get_hue_added_colors_list():
    hue_added_colors = []

    for i in range(360):
        image = np.array([[[255, 0, 0]]], dtype=np.uint8)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        image_hsv[:, :, 0] = i
        image_hsv[:, :, 1] = 1.0
        converted_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

        converted_image_int = list(converted_image.astype(np.uint8)[0][0][::-1])
        hue_added_colors.append(converted_image_int)

    return hue_added_colors


if __name__ == '__main__':
    hue_added_colors = get_hue_added_colors_list()
    for i in range(len(hue_added_colors)):
        print(i, hue_added_colors[i])
