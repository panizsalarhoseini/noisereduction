import cv2
import numpy as np

#ezafe kardan noise poisson
def AddingPoissonNoise(image):
    noisy_image = np.random.poisson(image / 255.0 * 60.0) / 60.0 * 255.0
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

#ezafe kardan noise uniform
def AddingUniformNoise(image, low=0, high=50):
    noise = np.random.uniform(low, high, image.shape).astype(np.float32)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

#reduction be vasile median
def reduce_noise_median(image):
    return cv2.medianBlur(image, 3)

#reduction be vasile gaussian
def reduce_noise_gaussian(image):
    return cv2.GaussianBlur(image, (5, 5), 1)

image = cv2.imread(r"Enter Image Path", cv2.IMREAD_GRAYSCALE)

noisy_image_poisson = AddingPoissonNoise(image)
noisy_image_uniform = AddingUniformNoise(image)
reduced_image_poisson = reduce_noise_median(noisy_image_poisson)
reduced_image_uniform = reduce_noise_gaussian(noisy_image_uniform)

cv2.imshow("OriginalImg",image)
cv2.imshow("NoisyImagePoisson",noisy_image_poisson)
cv2.imshow("NoisyImageUniform",noisy_image_uniform)
cv2.imshow("ReducedPoisson",reduced_image_poisson)
cv2.imshow("ReducedUniform",reduced_image_uniform)

cv2.waitKey(0)