import os
from PIL import Image

def convert_images(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of all files in the source folder
    files = os.listdir(source_folder)

    # Iterate over each file in the source folder
    for file in files:
        if file.endswith("fake_B.png"):
            # Generate the new filename by removing "fake_B" and changing the extension to ".jpg"
            new_filename = file.replace("fake_B", "").replace(".png", ".jpg")

            # Load the image
            image = Image.open(os.path.join(source_folder, file))

            # Convert and save the image in the destination folder
            image.save(os.path.join(destination_folder, new_filename))

            print(f"Converted {file} to {new_filename}")

# Specify the source and destination folders
source_folder = "/data1/liguanlin/codes/pytorch-CycleGAN-and-pix2pix/results/visdrone_day2night_cyclegan/test_latest/images"
destination_folder = "/data1/liguanlin/codes/pytorch-CycleGAN-and-pix2pix/datasets/visdrone_day2night/visdrone_day2night_cyclegan"

# Call the function to convert the images
convert_images(source_folder, destination_folder)