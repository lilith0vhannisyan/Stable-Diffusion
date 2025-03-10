# Text-Guided Image Style Transfer Using Stable Diffusion

This project implements a text-guided image style transfer system that leverages the Stable Diffusion v1-5 model with a DDIM scheduler for fast, deterministic inference. The application is built as an interactive web app using Streamlit, allowing users to upload an image, specify a text prompt (or select from pre-defined style templates), adjust various parameters, and generate a stylized version of the image. Additionally, the system computes the Structural Similarity Index (SSIM) to evaluate content preservation between the original and transformed images.

---

## Table of Contents

- [Abstract](#abstract)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters Explanation](#parameters-explanation)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Abstract

This report presents a system for text-guided image style transfer using the Stable Diffusion v1-5 model combined with a DDIM scheduler for fast and deterministic inference. Users can upload a relatively simple image—preferably one with minimal shapes and fewer particles to expedite processing—and specify a text prompt to guide the style transfer. By adjusting parameters such as style intensity (strength), number of inference steps, guidance scale, and output brightness, users can control the degree of stylistic transformation. The system also computes the SSIM between the original and stylized images to quantify content preservation. The complete code for this system is available in this repository.

---

## Features

- **Image-to-Image Style Transfer:** Upload an image and transform its style based on a text prompt.
- **Pre-defined Style Templates:** Choose from templates (Impressionist, Noir, Cyberpunk, Abstract) or use a custom prompt.
- **Adjustable Parameters:**  
  - Style Intensity (Strength)
  - Number of Inference Steps
  - Guidance Scale
  - Output Brightness
- **Evaluation:** Computes SSIM to assess content preservation.
- **Interactive Web Interface:** Built with Streamlit for an intuitive, real-time experience.

---

## Requirements

- **Python:** 3.8 or higher (Python 3.9+ recommended)
- **GPU (Optional):** Recommended for faster inference; CPU is supported but may be slower.
- **Dependencies:**
  - streamlit
  - diffusers
  - transformers
  - torch
  - pillow
  - torchvision
  - scikit-image
  - numpy

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
