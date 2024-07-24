import cv2, os, sys
import numpy as np


image1 = cv2.imread('log/Oxford102Flower/vqgan/run_2024-07-15-15-05-14/generated_images/vqgan_epoch049_5.jpg')
image2 = cv2.imread('log/Oxford102Flower/vqvae/run_2024-07-15-15-18-55/generated_images/vqvae_epoch049_5.jpg')

image1 = image1[2:, 2:-2]
image2 = image2[2:, 2:-2]

ori_imglist = []
img_list_1 = []
img_list_2 = []

step = 3
num_blocks = 6


for i in range(0, num_blocks*step, step):
    block = image1[i*258:(i+1)*258]
    block = block[:-2]
    h, w = block.shape[:2]
    block_1 = block[:, w//2:]
    ori_block = block[:, :w//2]
    ori_imglist.append(ori_block)
    img_list_1.append(block_1)

for i in range(0, num_blocks*step, step):
    block = image2[i*258:(i+1)*258]
    block = block[:-2]
    h, w = block.shape[:2]
    block_2 = block[:, w//2:]
    img_list_2.append(block_2)

color = (0, 0, 255)

ori_img = np.concatenate(ori_imglist, axis=1)
## draw title of "original" on the top left
cv2.putText(ori_img, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

img1 = np.concatenate(img_list_1, axis=1)
## draw title of "VQ-GAN" on the top left
cv2.putText(img1, 'VQ-GAN', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

img2 = np.concatenate(img_list_2, axis=1)
## draw title of "VQ-VAE" on the top left
cv2.putText(img2, 'VQ-VAE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

compose_img = np.concatenate([ori_img, img1, img2], axis=0)

cv2.imwrite('compose_img.jpg', compose_img)