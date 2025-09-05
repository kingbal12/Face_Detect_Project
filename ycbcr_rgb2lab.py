import math
import numpy as np

# R = 184
# G = 136
# B = 72

# #  Convert the RGB values to the YCbCr color space
# y = 0.299 * R + 0.587 * G + 0.114 * B
# cb = 0.564 * (B - y)
# cr = 0.713 * (R - y)

# # / Calculate the skin tone value based on YCbCr values
# skinTone = math.sqrt(math.pow(cb, 2) + math.pow(cr, 2))


# print(skinTone)


# brightness = [211, 193, 177] + 10

# print(brightness)


def rgb2lab(rgb):
    # RGB to XYZ
    rgb = rgb / 255.0
    mask = rgb > 0.04045
    rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] /= 12.92
    xyz = np.dot(
        rgb,
        np.array(
            [
                [0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227],
            ]
        ).T,
    )

    # XYZ to LAB
    xyz /= np.array([95.047, 100.0, 108.883])
    mask = xyz > 0.008856
    xyz[mask] = np.power(xyz[mask], 1 / 3)
    xyz[~mask] = (7.787 * xyz[~mask]) + (16 / 116)
    lab = np.zeros_like(xyz)
    lab[..., 0] = (116 * xyz[..., 1]) - 16
    lab[..., 1] = 500 * (xyz[..., 0] - xyz[..., 1])
    lab[..., 2] = 200 * (xyz[..., 1] - xyz[..., 2])

    return lab


rgb = [255, 255, 255]
lab = rgb2lab(rgb)
print(lab)
