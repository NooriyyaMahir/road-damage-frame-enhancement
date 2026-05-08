lap = cv2.Laplacian(gray, cv2.CV_64F)
sharpened = cv2.convertScaleAbs(gray - lap)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Before Sharpening")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(sharpened, cmap='gray')
plt.title("After Sharpening")
plt.axis('off')

plt.show()
