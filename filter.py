import cv2
import numpy as np


def mean_filter(image, k):
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.uint8)

    for i in range(k, height - k):
        for j in range(k, width - k):
            sum = 0
            for x in range(-k, k + 1):
                for y in range(-k, k + 1):
                    sum += image[i + x, j + y]
            output[i, j] = sum // ((2 * k + 1) ** 2)

    return output

def median_filter(image, k):
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.uint8)

    for i in range(k, height - k):
        for j in range(k, width - k):
            neighborhood = []
            for x in range(-k, k + 1):
                for y in range(-k, k + 1):
                    neighborhood.append(image[i + x, j + y])

if __name__ == "__main__":
    # 读取图像
    image = cv2.imread('/home/llj/code/test/data2/20230701/082405_ch01.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, None, fx=0.3, fy=0.3)

    # 调用均值滤波函数
    mean_filtered_image = mean_filter(image, 3)
    median_filtered_image = mean_filter(image, 3)

    # 显示原始图像和滤波后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Mean Filtered Image", mean_filtered_image)
    cv2.imshow("Median Filtered Image", median_filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # img = cv2.imread('/home/llj/code/test/data2/20230701/082405_ch01.jpg')
    # img = cv2.resize(img, None, fx=0.3, fy=0.3)
    # kernel = np.ones((5, 5), np.float32) / 25
    # dst = cv2.filter2D(img, -1, kernel)
    #
    # cv2.imshow("src", img)
    # cv2.imshow("filter2D", dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()