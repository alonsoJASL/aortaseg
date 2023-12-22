import os
import glob
import argparse
import nibabel as nib
import pandas as pd
import numpy as np


IMAGE_REGEX = "dcm-LGE-*"
AORTA = {"a" : "ascendingaorta.nii", "d": "descendingaorta.nii"}
def walk_sdirs(base_dir, number=-1, specific_case=None) -> list : 
    file_regex = IMAGE_REGEX

    subdirs = [x[0] for x in os.walk(base_dir)]
    filtered_subdirs = [d for d in subdirs if glob.glob(os.path.join(d, file_regex))]

    if specific_case is not None :
        filtered_subdirs = [d for d in filtered_subdirs if specific_case in d]
    
    if number > 0 :
        filtered_subdirs = filtered_subdirs[:number]

    return filtered_subdirs   

def extract_slices(input_folder, output_folder, csv_file, in_number=-1, in_specific_case=None):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    work_dirs = walk_sdirs(input_folder, number=in_number, specific_case=in_specific_case)

    # Create a DataFrame to store correspondences
    df = pd.DataFrame(columns=['image', 'mask'])
    
    # for wd in work_dirs :
    #     files_in_wd = os.listdir(wd)
    #     lge_files = [f for f in files_in_wd if f.startswith('dcm-LGE') and f.endswith('.nii')]

    #     for lge in lge_files : 
    #         img_path = os.path.join(wd, lge)
            


    # Get a list of all files in the input folder
    file_list = sorted(os.listdir(input_folder))


    for file in file_list:
        if file.endswith('.nii.gz'):
            # Load the NIfTI image
            img_path = os.path.join(input_folder, file)
            img = nib.load(img_path)
            img_data = img.get_fdata()

            # Iterate through axial slices
            for slice_idx in range(img_data.shape[2]):
                # Extract the 2D slice
                slice_data = img_data[:, :, slice_idx]

                # Define naming convention
                base_name = os.path.splitext(file)[0]
                image_name = f"{base_name}_slice_{slice_idx}.png"  # Adjust the extension as needed

                # Save the image
                image_path = os.path.join(output_folder, image_name)
                nib.save(nib.Nifti1Image(slice_data, img.affine), image_path)

                # Add correspondence to DataFrame
                df = df.append({'image': image_path, 'mask': ''}, ignore_index=True)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)

def main(args):
    # Call the extract_slices function
    extract_slices(args.image_folder, args.output_folder, args.csv_file)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract axial slices from 3D medical images.")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to the folder containing 3D medical images.")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the folder to save the axial slices.")
    parser.add_argument("--csv-file", type=str, required=True, help="Path to the CSV file for correspondences.")

    options_group = parser.add_argument_group("Options")
    options_group.add_argument("--number", type=int, default=-1, help="Number of subjects to process.")
    options_group.add_argument("--specific-case", type=str, default=None, help="Process a specific case.")
    args = parser.parse_args()

    main(args)
