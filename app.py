import os
import cv2
import numpy as np
import torch
from flask import Flask, request, render_template
from model import build_unet

app = Flask(__name__)

# Path to the checkpoint file
checkpoint_path = "files/savedModel/checkpoint.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model = build_unet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

def generate_segmentation(input_image_path, output_image_path):
    # Read the input image
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512, 512))  # Resize to match model input size
    x = np.transpose(image, (2, 0, 1))      
    x = x / 255.0
    x = np.expand_dims(x, axis=0)          
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    
    # Move input data to the same device as the model
    x = x.to(device)

    # Perform inference
    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0].cpu().numpy()        
        pred_y = np.squeeze(pred_y, axis=0)     
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

    # Create and save the output mask image
    pred_mask = mask_parse(pred_y)
    cv2.imwrite(output_image_path, pred_mask)

    return output_image_path 

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)

def adjust_contrast(image, contrast_factor):
    # Apply contrast reduction by multiplying pixel values by the factor
    adjusted_image = (image * 0.9).astype(np.uint8)
    return adjusted_image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the uploaded file
            input_image_path = "uploads/input_image.jpg"
            file.save(input_image_path)

            # Generate segmentation
            output_image_path = "static/output/output_mask.png"
            segmented_image_path = generate_segmentation(input_image_path, output_image_path)

            # Process the segmented image
            img = cv2.imread(segmented_image_path)

            # Apply gamma correction
            gamma = 50.0
            img_gamma_corrected = adjust_gamma(img, gamma)

            # Convert to grayscale
            img_gray = cv2.cvtColor(img_gamma_corrected, cv2.COLOR_BGR2GRAY)

            # Apply contrast reduction
            img_low_contrast = (img_gray * 0.9).astype(np.uint8)

            # Save the processed image
            processed_image_path = "static/output/processed_image.jpg"
            cv2.imwrite(processed_image_path, img_low_contrast)

            # Render the result page with image paths
            return render_template('result.html', 
                                   input_image="uploads/input_image.jpg", 
                                   segmented_image=segmented_image_path, 
                                    processed_image="static/output/processed_image.jpg")

    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
