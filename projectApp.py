import streamlit as st
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
import numpy as np
from skimage.metrics import structural_similarity as ssim

@st.cache_resource
def load_pipeline():
    #st.write("Debug: Loading pipeline...")
    # Load the pre-trained Stable Diffusion Img2Img pipeline using runwayml/stable-diffusion-v1-5.
    # Use float16 if GPU is available; otherwise, load with default dtype.
    if torch.cuda.is_available():
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16,
        )
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #st.write("Debug: Using device:", device)
    pipe = pipe.to(device)
    
    # Replace the default scheduler with the DDIM scheduler for faster, deterministic inference.
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    #st.write("Debug: Pipeline loaded successfully with DDIM scheduler!")
    return pipe

# Load the pipeline (cached for efficiency)
pipe = load_pipeline()

# Application Title and Description
st.title("Text-Guided Image Style Transfer")
st.write("""
This app uses the Stable Diffusion v1-5 model with a DDIM scheduler for text-guided image-to-image style transfer.
You can adjust key parameters:
- **Number of Inference Steps:** Controls how many denoising steps are taken (range: 10 to 100). More steps may improve quality but slow down inference.
- **Guidance Scale:** Determines how strongly the model follows your text prompt (range: 1.00 to 10.00). Higher values enforce closer adherence at the potential cost of image quality.
- **Output Brightness:** (Optional) Adjust the brightness of the final image (range: 0.5 to 1.5). A factor of 1.0 means no change.
""")

# Sidebar: Style Template Selection
st.sidebar.header("Style Templates")
style_templates = {
    "Custom": "",
    "Impressionist": "A painting in the style of Monet, with soft brush strokes and vibrant colors.",
    "Noir": "A high-contrast, black-and-white scene with dramatic shadows and a mysterious atmosphere.",
    "Cyberpunk": "A futuristic, neon-lit cityscape with a gritty, dystopian vibe.",
    "Abstract": "An abstract painting with bold colors and dynamic shapes, evoking a sense of motion."
}
selected_style = st.sidebar.selectbox("Select a style template", list(style_templates.keys()))

# Prompt Input: Use pre-defined template or allow custom input.
if selected_style == "Custom":
    prompt = st.text_input("Enter your custom style prompt", "Describe the desired style...")
else:
    default_prompt = style_templates[selected_style]
    prompt = st.text_area("Style Prompt (feel free to modify)", default_prompt)

# Additional Hyperparameter Controls for Inference
num_inference_steps = st.slider(
    "Number of Inference Steps", 
    min_value=10, max_value=100, value=50,
    help="The number of denoising steps (more steps may improve quality but increase computation time)."
)
guidance_scale = st.slider(
    "Guidance Scale", 
    min_value=1.0, max_value=10.0, value=7.5,
    help="Higher values cause the model to follow the prompt more strictly."
)
brightness_factor = st.slider(
    "Output Brightness", 
    min_value=0.5, max_value=1.5, value=1.0,
    help="Adjust the brightness of the final output image (1.0 means no change)."
)

# Image Upload Section
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    #st.write("Debug: File uploaded.")
    try:
        init_image = Image.open(uploaded_file).convert("RGB")
        #st.write("Debug: Successfully opened image.")
    except Exception as e:
        st.error("Error opening image: " + str(e))
    
    # st.write("Debug: Image type:", type(init_image))
    # st.write("Debug: Image mode:", init_image.mode)
    # st.write("Debug: Original image size:", init_image.size)
    
    # Resize the image to 512x512 (commonly expected by the model)
    init_image = init_image.resize((512, 512))
    #st.write("Debug: Resized image size:", init_image.size)
    
    st.image(init_image, caption="Original Image", use_container_width=True)
    
    # Convert the PIL image to a torch.Tensor using transforms.ToTensor()
    to_tensor = transforms.ToTensor()
    init_image_tensor = to_tensor(init_image)  # shape: (3, 512, 512), dtype: torch.float32
    #st.write("Debug: Tensor shape:", init_image_tensor.shape, "dtype:", init_image_tensor.dtype)
    
    # Add a batch dimension: (1, 3, 512, 512)
    init_image_tensor = init_image_tensor.unsqueeze(0)
    #st.write("Debug: Tensor shape after unsqueeze:", init_image_tensor.shape)
    
    # Move tensor to the appropriate device; if GPU available, convert to float16, otherwise keep float32.
    if torch.cuda.is_available():
        init_image_tensor = init_image_tensor.to(pipe.device).to(torch.float16)
    else:
        init_image_tensor = init_image_tensor.to(pipe.device)
    #st.write("Debug: Final tensor dtype:", init_image_tensor.dtype)
else:
    st.info("Please upload an image to start.")

#st.write("Debug: Prompt:", prompt)
#st.write("Debug: Strength (Style Intensity):", st.session_state.get("strength", None))

# Style Intensity Slider (used as the strength parameter)
strength = st.slider(
    "Style Intensity (Strength)",
    min_value=0.0, max_value=1.0, value=0.3,  # Default changed from 0.75 to 0.3 for demonstration
    help="0.0 means no style change; 1.0 means maximum style transformation."
)
#st.write("Debug: Updated Strength:", strength)

# Generate Button: Process the image when clicked.
if st.button("Generate Styled Image"):
    if not uploaded_file:
        st.error("Please upload an image before generating.")
    else:
        with st.spinner("Applying style transfer..."):
            try:
                # Call the pipeline with additional parameters and using DDIM scheduler for inference.
                result = pipe(
                    prompt=prompt, 
                    image=init_image_tensor, 
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
                #st.write("Debug: Style transfer completed successfully.")
            except Exception as e:
                st.error("Error during style transfer: " + str(e))
                #st.write("Debug: Input tensor details:")
                st.write("Type:", type(init_image_tensor))
                st.write("Shape:", init_image_tensor.shape)
                raise e
        
        # Optional post-processing: Adjust brightness of the output image.
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(brightness_factor)
            #st.write("Debug: Applied brightness enhancement with factor:", brightness_factor)
        
        # Display the final styled image.
        st.image(result, caption="Styled Image", use_container_width=True)
        
        # Evaluate the result: Compute SSIM between the original and styled image (grayscale comparison).
        original_np = np.array(init_image.convert("L"))
        result_np = np.array(result.convert("L"))
        ssim_index = ssim(original_np, result_np)
        st.write("Evaluation: SSIM between original and styled image =", round(ssim_index, 4))
