import cv2
import numpy as np
from math import sqrt, floor
import matplotlib.pyplot as plt
from PIL import Image

def read_image(path):
    img = cv2.imread(path)  # cv2.IMREAD_GRAYSCALE)
    size = img.shape
    dimension = (size[0], size[1])

    return img, size, dimension

def image_change_scale(img, dimension, scale=100, interpolation=cv2.INTER_LINEAR):
    scale /= 100
    new_dimension = (int(dimension[1]*scale), int(dimension[0]*scale))
    resized_img = cv2.resize(img, new_dimension, interpolation=interpolation)

    return resized_img

def nearest_interpolation(image, dimension):
    h_out, w_out = dimension
    h_in, w_in = image.shape[:2]
    new_image = np.zeros((h_out, w_out, image.shape[2]), dtype=image.dtype)

    scale_y = h_in / h_out
    scale_x = w_in / w_out

    for i in range(h_out):
        for j in range(w_out):
            row = int(i * scale_y)
            col = int(j * scale_x)
            new_image[i, j] = image[row, col]

    return new_image

def nearest_interpolation_vectorized(image, dimension):
    h_out, w_out = dimension
    h_in, w_in = image.shape[:2]

    scale_y = h_in / h_out
    scale_x = w_in / w_out

    i = np.arange(h_out)
    j = np.arange(w_out)
    jj, ii = np.meshgrid(j, i) 

    row = (ii * scale_y).astype(int)
    col = (jj * scale_x).astype(int)

    return image[row, col]

def bilinear_interpolation(image, dimension):
    height = image.shape[0]
    width = image.shape[1]

    scale_x = (width)/(dimension[1])
    scale_y = (height)/(dimension[0])

    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    for k in range(3):
        for i in range(dimension[0]):
            for j in range(dimension[1]):
                x = (j+0.5) * (scale_x) - 0.5
                y = (i+0.5) * (scale_y) - 0.5

                x_int = int(x)
                y_int = int(y)

                x_int = min(x_int, width-2)
                y_int = min(y_int, height-2)

                x_diff = x - x_int
                y_diff = y - y_int

                a = image[y_int, x_int, k]
                b = image[y_int, x_int+1, k]
                c = image[y_int+1, x_int, k]
                d = image[y_int+1, x_int+1, k]

                pixel = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
                    (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff

                new_image[i, j, k] = pixel.astype(np.uint8)

    return new_image

def main():
    images_list = {}

    # Read Image
    img, size, dimension = read_image("./images/planet.jpg")
    print(f"Image size is: {size}")
    images_list['Original Image'] = img

    # Change Image Size
    scale_percent = 25  # percent of original image size
    resized_img = image_change_scale(img, dimension, scale_percent)
    print(f"Smalled Image size is: {resized_img.shape}")
    images_list['Smalled Image'] = resized_img

    # Nearest Interpolation
    nn_img_algo = nearest_interpolation(resized_img, dimension)
    nn_img_algo = cv2.cvtColor(np.array(nn_img_algo), cv2.COLOR_BGR2RGB)
    nn_img_algo = Image.fromarray(nn_img_algo.astype('uint8'))
    nn_img_algo.save("./images/my_nearest.png")

    nn_img = image_change_scale(resized_img, dimension, interpolation=cv2.INTER_NEAREST)
    nn_img = cv2.cvtColor(np.array(nn_img), cv2.COLOR_BGR2RGB)
    nn_img = Image.fromarray(nn_img.astype('uint8'))
    nn_img.save("./images/opencv_nearest.png")

    fig1, axs1 = plt.subplots(1, 4)
    # fig.suptitle('25 Percent of the original size', fontsize=16)
    axs1[0].set_title('Original 256x256 Image')
    axs1[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs1[1].set_title('64x64 Image')
    axs1[1].imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    axs1[2].set_title('256x256 Image with Nearest')
    axs1[2].imshow(nn_img_algo)
    axs1[3].set_title('256x256 Image with OpenCV\'s Nearest')
    axs1[3].imshow(nn_img)

    # Bilinear Interpolation
    bl_img_algo = bilinear_interpolation(resized_img, dimension)
    bl_img_algo = cv2.cvtColor(np.array(bl_img_algo, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    bl_img_algo = Image.fromarray(bl_img_algo.astype('uint8'))
    bl_img_algo.save("./images/my_bilinear.png")

    bl_img = image_change_scale(resized_img, dimension, interpolation=cv2.INTER_LINEAR)
    bl_img = cv2.cvtColor(np.array(bl_img), cv2.COLOR_BGR2RGB)
    bl_img = Image.fromarray(bl_img.astype('uint8'))
    bl_img.save("./images/opencv_bilinear.png")

    fig2, axs2 = plt.subplots(1, 4)
    # fig.suptitle('25 Percent of the original size', fontsize=16)
    axs2[0].set_title('Original 256x256 Image')
    axs2[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs2[1].set_title('64x64 Image')
    axs2[1].imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    axs2[2].set_title('256x256 Image with Bilinear')
    axs2[2].imshow(bl_img_algo)
    axs2[3].set_title('256x256 Image with OpenCV\'s Bilinear')
    axs2[3].imshow(bl_img)
    plt.show()

if __name__ == "__main__":
    main()