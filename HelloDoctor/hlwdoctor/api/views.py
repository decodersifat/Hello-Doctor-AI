from django.shortcuts import render
from django.http import HttpResponse 
import nibabel as nib
import glob
import numpy as np
import tempfile
import os
import io
from django.conf import settings
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import datetime  
import base64
from session.models import Profile
import numpy as np
from scipy.ndimage import zoom
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model 
from matplotlib import pyplot as plt
from django.http import FileResponse, Http404
import random


def mri_input_form(request):
    return render(request, "mri_input/mri_from.html")




def predict(request):
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('image')
        if len(uploaded_files) != 4:
            return HttpResponse("Please upload exactly 4 files.")

        flair_file, t1ce_file, t2_file, mask_file = uploaded_files
        
        profile = Profile.objects.get(user=request.user)


        first_name = profile.user.first_name
        last_name = profile.user.last_name
        age = profile.age
        

        # Preprocess / model prediction:
        img, mask = preprocessing(flair_file, t1ce_file, t2_file, mask_file)
        test_img_input = np.expand_dims(img, axis=0)
        model = load_model(r"D:\Hello_Doctor_AI_Diagnostic_Center\AI developmnet\sdprojectJun\sdproject\hlwdoctor\api\model-V2-diceLoss_focal.keras", compile=False)
        test_prediction = model.predict(test_img_input)
        
        test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

        affine = np.eye(4)
        final_mask = nib.Nifti1Image(test_prediction_argmax.astype(np.int32), affine)  
        

        analysis_output = calculate(final_mask)
        output_pdf_path = os.path.join(tempfile.gettempdir(), "MRI_Report.pdf")

        generate_pdf(output_pdf_path, analysis_output, first_name, last_name , age, img, test_prediction_argmax )
        print("PDF generated successfully!")


        n_slice = np.random.randint(0, test_prediction_argmax.shape[2])
        # --- 1) Create plot in memory ---
        fig, ax = plt.subplots()
        ax.imshow(test_prediction_argmax[:, :, n_slice], cmap='viridis')
        ax.set_title("Prediction on test image")
        ax.axis('off') 

        # --- 2) Save figure to a BytesIO buffer ---
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        plt.close(fig)  # Close the figure to release memory
        buffer.seek(0)

        # --- 3) Encode plot to base64 string ---
        image_png = buffer.getvalue()
        base64_string = base64.b64encode(image_png).decode('utf-8')

        # --- 4) Pass base64 string to an HTML template ---
        return render(request, 'results.html', {
            'plot_base64': base64_string,
            'pdf_url': f'/download_pdf/{os.path.basename(output_pdf_path)}',
        })
    else:
        return HttpResponse("Invalid request method.")



def preprocessing(flair_file, t1ce_file, t2_file, mask_file):
    
    flair_path = save_to_temp_file(flair_file)
    t1ce_path  = save_to_temp_file(t1ce_file)
    t2_path    = save_to_temp_file(t2_file)
    mask_path  = save_to_temp_file(mask_file)

    # 2) Now load from the saved paths
    flair = nib.load(flair_path).get_fdata()
    t1ce  = nib.load(t1ce_path).get_fdata()
    t2    = nib.load(t2_path).get_fdata()
    mask  = nib.load(mask_path).get_fdata()

    scaler = MinMaxScaler()

    #Scalers are applied to 1D so let us reshape and then reshape back to original shape. 
    test_image_flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)
    test_image_t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)
    test_image_t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)
    test_mask = mask.astype(np.uint8)
    
    test_mask[test_mask==4] = 3  
    combined_img = np.stack([test_image_flair,test_image_t1ce,test_image_t2], axis = 3)
    mask = test_mask
    combined_img = combined_img[40:210, 40:210, :]
    mask = mask[40:210, 40:210, :]
    # Define the desired target shape
    target_shape = (128, 128, 128)
    # Calculate zoom factors for the combined image
    zoom_factors_img = tuple(t / o for t, o in zip(target_shape, combined_img.shape[:3]))
    # Resize combined image using cubic interpolation
    resized_combined_img = zoom(combined_img, zoom_factors_img + (1,), order=3)  # Keep channels intact
    # Calculate zoom factors for the mask
    zoom_factors_mask = tuple(t / o for t, o in zip(target_shape, mask.shape))
    # Resize mask using nearest-neighbor interpolation
    resized_mask = zoom(mask, zoom_factors_mask, order=0).astype(np.uint8)  # Ensure mask is integer
    
    return resized_combined_img, resized_mask








def calculate(final_mask):
    voxel_dimensions = final_mask.header.get_zooms()
    voxel_volume = np.prod(voxel_dimensions)  # Volume of a single voxel in mm³
    seg_data = final_mask.get_fdata()

    # Step 3: Count the number of voxels for each label
    labels, voxel_counts = np.unique(seg_data, return_counts=True)

    # Step 4: Calculate volume for each label
    label_volumes = {
        label: count * voxel_volume for label, count in zip(labels, voxel_counts)
    }

    # Step 5: Calculate the total tumor volume (excluding label 0)
    total_tumor_volume = sum(volume for label, volume in label_volumes.items() if label != 0)

    # Step 6: Calculate the percentage of each label
    label_percentages = {
        label: (volume / total_tumor_volume * 100) if label != 0 else 0
        for label, volume in label_volumes.items()
    }

    # Step 7: Prepare the results as a string
    results = []
    results.append("\n\n------------------------------------------------------------------------------------------------------------------------")
    results.append(f"Voxel Dimensions: {voxel_dimensions} mm")
    results.append(f"Voxel Volume: {voxel_volume:.2f} mm³")
    results.append("------------------------------------------------------------------------------------------------------------------------")
    


    results.append("\n\nTumor Volumes States (in mm³):")
    results.append("------------------------------------------------------------------------------------------------------------------------")
    label_item = [
        "Blank Space or No Tumor (BT)",
        "Necrotic and Non-enhancing tumor core(NCR/NET)",
        "Peritumoral edema(ED)",
        "Enhancing tumor (ET)"
    ]

    for label, volume in label_volumes.items():
        if label < len(label_item):
            results.append(f" {label_item[int(label)]}: {volume:.2f} mm³")
        else:
            results.append(f" {label}: {volume:.2f} mm³")  

    results.append("\n\nTumor Volumes State Percentages (of total tumor volume):")
    results.append("------------------------------------------------------------------------------------------------------------------------")

    
    for label, percentage in label_percentages.items():


        if label != 0 and label < len(label_item) :  # Exclude background from percentage display
            results.append(f" {label_item[int(label)]}: {percentage:.2f} %")
    results.append("\n\n------------------------------------------------------------------------------------------------------------------------")
    results.append(f"Total Tumor Volume (excluding Blank Space or No Tumor (BT)): {total_tumor_volume:.2f} mm³\n")
    # Return the results as a single formatted string
    return "\n".join(results)





def generate_pdf(output_filename, analysis_output, first_name, last_name, age, img ,test_prediction_argmax):
    # Use settings.MEDIA_ROOT for storing files.
    output_dir = os.path.join(settings.MEDIA_ROOT, "reports")
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    output_filename = os.path.join(output_dir, "MRI_Report.pdf")

    # Create a canvas
    pdf_canvas = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter

    # Add company logo (top-right)
    logo_path = r"D:\Hello_Doctor_AI_Diagnostic_Center\AI developmnet\sdprojectJun\sdproject\hlwdoctor\static\images\rivers_20241124_193633_0000.png"
    if os.path.exists(logo_path):
        logo = ImageReader(logo_path)
        pdf_canvas.drawImage(logo, width - 175, height - 160, width=180, height=190, mask='auto')

    # Add generated date (top-left)
    pdf_canvas.setFont("Helvetica", 10)
    generate_date = f"Generated Date: {datetime.date.today()}"
    pdf_canvas.drawString(50, height - 50, generate_date)

    # Add heading
    pdf_canvas.setFont("Helvetica-Bold", 16)
    pdf_canvas.drawString(50, height - 120, "MRI Report")

    # Add patient information
    pdf_canvas.setFont("Helvetica", 12)
    patient_info = [
        f"Name: {first_name} {last_name}",
        f"Age: {age}",
    ]

    # Add analysis output to patient info
    analysis_lines = analysis_output.split("\n")
    patient_info.extend(analysis_lines)  # Append the analysis lines to patient_info

    # Write patient information to PDF
    y_position = height - 150
    for info in patient_info:
        pdf_canvas.drawString(50, y_position, info)
        y_position -= 20
        if y_position < 50:  # Create a new page if space is insufficient
            pdf_canvas.showPage()
            pdf_canvas.setFont("Helvetica", 12)
            y_position = height - 50

    # Page 2: Create images with 3 columns and dynamic rows
    # Dimensions for the grid
    images_per_row = 3  # 3 columns
    image_width = (width - 100) / images_per_row  # Leave margins
    image_height = 150  # Fixed height for each image
    x_margin = 50
    y_margin = 100  # Margin for title and spacing

    # Title font
    title = "Tumor Analysis Report"
    pdf_canvas.setFont("Helvetica-Bold", 16)

    # Calculate available rows per page
    rows_per_page = int((height - y_margin - 50) / (image_height + 20))  # Subtract margins and spacing

    # Find valid slices (those with labels 1, 2, or 3 in the mask)
    valid_slices = []
    for slice_index in range(test_prediction_argmax.shape[2]):
        if np.any(np.isin(test_prediction_argmax[:,:,slice_index], [1, 2, 3])):  # Contains labels 1, 2, or 3
            valid_slices.append(slice_index)

    # Generate and add images in a grid layout
    for slice_index in range(len(valid_slices)):
        if slice_index % (rows_per_page * images_per_row) == 0:  # New page
            pdf_canvas.showPage()
            pdf_canvas.setFont("Helvetica-Bold", 16)

            # Add title to the top of the page
            title_width = pdf_canvas.stringWidth(title, "Helvetica-Bold", 16)
            pdf_canvas.drawString((width - title_width) / 2, height - 50, title)

        # Determine the row and column for the current image
        page_index = slice_index % (rows_per_page * images_per_row)
        row = page_index // images_per_row
        col = page_index % images_per_row

        # Calculate the x and y position for the image
        x_position = x_margin + col * image_width
        y_position = height - y_margin - (row * (image_height + 20)) - image_height

        # Use the valid slice for plotting
        n_slice = valid_slices[slice_index]  # Select the slice with label 1, 2, or 3
        fig, ax = plt.subplots()
        ax.imshow(img[:, :, n_slice, 0], cmap="gray") 
        ax.imshow(test_prediction_argmax[:, :, n_slice],  alpha=0.5, cmap='coolwarm')
        ax.set_title(f"Slice {n_slice + 1}")
        ax.axis('off')

        # Save the plot as a temporary image
        temp_image_path = f"temp_plot_{slice_index}.png"
        plt.savefig(temp_image_path)
        plt.close()

        # Add the image to the PDF
        pdf_canvas.drawImage(temp_image_path, x_position, y_position, width=image_width, height=image_height, mask='auto')

        # Clean up the temporary image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    # Save and close the PDF
    pdf_canvas.save()





def save_to_temp_file(uploaded_file):
    """Write an UploadedFile or TemporaryUploadedFile to disk and return the file path."""
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        tmp.flush()
        return tmp.name 
    




def download_pdf(request, filename):
    # Construct the full file path
    file_path = os.path.join(settings.MEDIA_ROOT, 'reports', filename)
    if os.path.exists(file_path):
        try:
            # Open the file without using a with-statement
            file = open(file_path, 'rb')
            return FileResponse(file, as_attachment=True, filename=filename)
        except PermissionError:
            raise Http404("File is currently being used by another process.")
    else:
        raise Http404("File not found.")