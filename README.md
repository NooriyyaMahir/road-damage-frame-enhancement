# Road Damage Detection – Image Enhancement Pipeline

## Project Overview

This project focuses on enhancing road video frames to improve the detection of road surface damages such as cracks and potholes. Raw frames captured using mobile cameras often contain noise, uneven illumination, and motion blur. These degradations can reduce the accuracy of segmentation and damage detection.

The proposed system applies an image enhancement pipeline to improve frame quality before performing further analysis.

---

## Enhancement Pipeline

The implemented image enhancement pipeline consists of the following stages:

1. Frame Extraction from Video
2. Grayscale Conversion
3. Noise Removal
4. Contrast Enhancement
5. Image Sharpening

Pipeline Flow:

Original Frame  
↓  
Grayscale Conversion  
↓  
Noise Removal  
↓  
Contrast Enhancement  
↓  
Image Sharpening  
↓  
Enhanced Frame

---

## Technologies Used

Python

OpenCV – image processing  
NumPy – numerical computation  
Matplotlib – visualization  
PyWavelets – wavelet transform  
SciPy – statistical analysis

---

## Project Structure
