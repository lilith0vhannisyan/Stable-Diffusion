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

2. **Install Dependencies:**

   ```bash
   pip install --upgrade streamlit diffusers transformers torch pillow torchvision scikit-image numpy

---

## Usage
1. **Run the Streamlit App**
    ```bash
   streamlit run projectApp.py

---

## In the Web Interface

### Upload an Image
- **Action:** Use the uploader to select an image file (PNG, JPG, JPEG).
- **Tip:** Choose a simple image with minimal shapes and fewer particles for faster processing.

### Select or Enter a Prompt
- **Action:** Use a pre-defined style template or enter a custom prompt.
- **Example:** For quick processing, you might use an empty prompt `" "`.

### Adjust Parameters
- **Style Intensity (Strength):**
  - **Range:** 0.0 to 1.0.
  - **Example:** A strength of 0.69 applies moderate transformation, preserving most of the original image.
  - **Impact:** Lower values (e.g., 0.3) preserve more of the original content, while higher values introduce stronger stylistic effects.
- **Number of Inference Steps:**
  - **Range:** 10 to 100.
  - **Impact:** More steps may yield higher quality images but take longer to process.
- **Guidance Scale:**
  - **Range:** 1.0 to 10.0.
  - **Impact:** Higher values force the model to follow the text prompt more strictly, potentially improving prompt adherence.
- **Output Brightness:**
  - **Range:** 0.5 to 1.5.
  - **Impact:** Adjusts the brightness of the final image; 1.0 means no change.

### Generate
- **Action:** Click the **"Generate Styled Image"** button to run the style transfer.

### View Results
- **Outcome:** The app displays:
  - The original image.
  - The transformed (styled) image.
  - Evaluation metrics (e.g., SSIM) that indicate how well the original content is preserved.

---

## Parameters Explanation

- **Style Intensity (Strength):**
  - **Range:** 0.0 (no change) to 1.0 (maximum transformation).
  - **Impact:**  
    - Lower values (e.g., 0.3) preserve more of the original image content.
    - Higher values (e.g., 0.69) apply stronger transformations, adding more noise and stylistic effects.
- **Number of Inference Steps:**
  - **Range:** 10 to 100.
  - **Impact:** Increasing the number of steps can enhance image quality, though at the cost of longer processing times.
- **Guidance Scale:**
  - **Range:** 1.0 to 10.0.
  - **Impact:** Higher values cause the model to adhere more strictly to the text prompt, potentially improving the output's relevance to the prompt.
- **Output Brightness:**
  - **Range:** 0.5 to 1.5.
  - **Impact:** Alters the brightness of the final image, where 1.0 indicates no change.

---

## Troubleshooting

- **Device Compatibility:**  
  The model loads in `float16` when a GPU is available for faster inference. On a CPU, it defaults to `float32`. If you encounter issues, ensure your device settings are correct.
- **Dependency Issues:**  
  Ensure all required packages are installed and up to date.
- **Error Messages:**  
  Refer to the debug prints in the app for guidance on any issues with image conversion or model inference.

---

## References

- **ChatGPT (OpenAI, 2023).**
- **Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.**
- **Song, Y., et al. Denoising Diffusion Implicit Models.**
- **Radford, A., et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP).**
- **[Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers).**
- **[Streamlit Documentation](https://docs.streamlit.io).**
- **[Python Official Documentation](https://docs.python.org).**
- **[How to Come Up with Good Prompts for AI Image Generation](https://stable-diffusion-art.com/how-to-come-up-with-good-prompts-for-ai-image-generation/).**
- **[Nature Article on Diffusion Models](https://www.nature.com/articles/s41598-023-39278-0).**
