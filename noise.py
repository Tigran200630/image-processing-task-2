# Экспоненциальный шум: p(z) = a * exp(-a * z) при z >= 0, иначе 0 (лекция 2, слайд "Экспоненциальный шум").
# Среднее шума равно 1 / a. Модель шума: f = s + n.
import numpy as np


def exponential_noise(shape, a):
    # Метод обратной функции: F(z) = 1 - exp(-a * z)  =>  z = -ln(1 - U) / a, где U равномерно на [0, 1)
    u = np.random.uniform(0, 1, shape)
    return -np.log(1 - u) / a


def add_exponential_noise(img, a):
    # f = s + n, результат обрезаем до [0, 255]. Шум только положительный, поэтому картинка светлеет в среднем на 1 / a.
    noisy = img.astype(np.float64) + exponential_noise(img.shape, a)
    return np.clip(noisy, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import cv2

    img = cv2.imread("sar_1.jpg", cv2.IMREAD_GRAYSCALE)
    noisy = add_exponential_noise(img, a=1 / 30)
    print("средняя яркость: %.1f -> %.1f" % (img.mean(), noisy.mean()))
