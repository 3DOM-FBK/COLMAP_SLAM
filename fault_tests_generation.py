import shutil
import os
from pathlib import Path
import numpy as np
from PIL import Image

original_images_dir = "/home/threedom/Desktop/github_3dom/COLMAP_SLAM/_DATA/CARLA/images_original"
output_dir = "/home/threedom/Desktop/github_3dom/COLMAP_SLAM/_DATA/CARLA/images"
fault_mode = "cam1_repeat_black"  # Options: "missing", "corrupt", "duplicate", "normal", "cam0_repeat", "cam1_repeat", "cam0_repeat_black", "cam1_repeat_black"

# Image range selection
start_index = 0      # Starting index (inclusive)
end_index = 50       # Ending index (exclusive, -1 means all images)

# Camera repeat fault configuration
repeat_start_index = 20    # Index where the specified camera starts repeating images
repeat_source_index = 19   # Index of the image that will be repeated (not used for black modes)

# Check if output directory exists and is empty
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
elif os.listdir(output_dir):
    print(f"Output directory {output_dir} is not empty. Please clear it before running the script.")
    exit(1)

class GenerateData:
    def __init__(self, original_images_dir, output_dir, fault_mode, start_index=0, end_index=-1, 
                 repeat_start_index=20, repeat_source_index=15):
        self.original_images_dir = original_images_dir
        self.output_dir = output_dir
        self.fault_mode = fault_mode
        self.start_index = start_index
        self.end_index = end_index
        self.repeat_start_index = repeat_start_index
        self.repeat_source_index = repeat_source_index
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    def get_image_dimensions(self, image_path):
        """Get dimensions of an image file"""
        try:
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except Exception as e:
            print(f"Warning: Could not read image {image_path}: {e}")
            return (640, 480)  # Default fallback dimensions

    def create_black_image(self, reference_path, output_path):
        """Create a black image with the same dimensions as reference image"""
        try:
            width, height = self.get_image_dimensions(reference_path)
            
            # Create black image
            black_image = Image.new('RGB', (width, height), (0, 0, 0))
            
            # Save with same format as reference
            black_image.save(output_path)
            return True
        except Exception as e:
            print(f"Error creating black image: {e}")
            return False

    def get_images_by_subfolder(self):
        """Get images organized by subfolder to maintain structure"""
        subfolder_images = {}
        
        for root, dirs, files in os.walk(self.original_images_dir):
            current_path = Path(root)
            
            # Get relative path from original_images_dir
            if current_path == Path(self.original_images_dir):
                subfolder_name = "root"
            else:
                subfolder_name = current_path.relative_to(self.original_images_dir).parts[0]
            
            if subfolder_name not in subfolder_images:
                subfolder_images[subfolder_name] = []
            
            for file in files:
                file_path = current_path / file
                if file_path.suffix.lower() in self.supported_extensions:
                    relative_path = file_path.relative_to(self.original_images_dir)
                    subfolder_images[subfolder_name].append(relative_path)
        
        # Sort images within each subfolder
        for subfolder in subfolder_images:
            subfolder_images[subfolder] = sorted(subfolder_images[subfolder])
        
        return subfolder_images

    def ensure_output_dir_structure(self, relative_path):
        """Create the necessary subdirectory structure in output directory"""
        output_subdir = Path(self.output_dir) / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        return output_subdir

    def validate_indices(self, total_images):
        """Validate and adjust start/end indices"""
        if self.start_index < 0:
            self.start_index = 0
        
        if self.end_index == -1 or self.end_index > total_images:
            self.end_index = total_images
        
        if self.start_index >= self.end_index:
            raise ValueError(f"Invalid indices: start_index ({self.start_index}) must be less than end_index ({self.end_index})")
        
        if self.start_index >= total_images:
            raise ValueError(f"start_index ({self.start_index}) is greater than or equal to total images ({total_images})")
        
        # Validate repeat indices for camera repeat modes
        if self.fault_mode in ["cam0_repeat", "cam1_repeat"]:
            if self.repeat_start_index >= self.end_index:
                raise ValueError(f"repeat_start_index ({self.repeat_start_index}) must be less than end_index ({self.end_index})")
            if self.repeat_source_index >= self.repeat_start_index:
                raise ValueError(f"repeat_source_index ({self.repeat_source_index}) must be less than repeat_start_index ({self.repeat_start_index})")
            if self.repeat_source_index < self.start_index:
                raise ValueError(f"repeat_source_index ({self.repeat_source_index}) must be >= start_index ({self.start_index})")
        
        # Validate repeat indices for camera black repeat modes
        elif self.fault_mode in ["cam0_repeat_black", "cam1_repeat_black"]:
            if self.repeat_start_index >= self.end_index:
                raise ValueError(f"repeat_start_index ({self.repeat_start_index}) must be less than end_index ({self.end_index})")

    def generate(self):
        """Generate test data with various fault modes"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Get images organized by subfolder
        subfolder_images = self.get_images_by_subfolder()
        
        if not subfolder_images:
            print(f"No image files found in {self.original_images_dir}")
            return

        print("Found subfolders and their image counts:")
        total_images = 0
        for subfolder, images in subfolder_images.items():
            print(f"  {subfolder}: {len(images)} images")
            total_images += len(images)

        # Get unique filenames (timestamps) to apply range selection
        unique_filenames = set()
        for images in subfolder_images.values():
            for img_path in images:
                unique_filenames.add(img_path.name)
        
        unique_filenames = sorted(unique_filenames)
        
        # Validate indices based on unique filenames
        try:
            self.validate_indices(len(unique_filenames))
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Select filenames based on indices
        selected_filenames = unique_filenames[self.start_index:self.end_index]
        
        print(f"\nFound {len(unique_filenames)} unique timestamps")
        print(f"Processing timestamps from index {self.start_index} to {self.end_index-1} ({len(selected_filenames)} timestamps)")
        print(f"Fault mode: {self.fault_mode}")
        
        if self.fault_mode in ["cam0_repeat", "cam1_repeat"]:
            camera_name = "cam0" if self.fault_mode == "cam0_repeat" else "cam1"
            print(f"{camera_name.upper()} repeat configuration:")
            print(f"  - Normal images: indices {self.start_index} to {self.repeat_start_index-1}")
            print(f"  - Repeated image: index {self.repeat_source_index} ('{selected_filenames[self.repeat_source_index - self.start_index]}')")
            print(f"  - Repeat starts at: index {self.repeat_start_index}")
        
        elif self.fault_mode in ["cam0_repeat_black", "cam1_repeat_black"]:
            camera_name = "cam0" if self.fault_mode == "cam0_repeat_black" else "cam1"
            print(f"{camera_name.upper()} black repeat configuration:")
            print(f"  - Normal images: indices {self.start_index} to {self.repeat_start_index-1}")
            print(f"  - Black images start at: index {self.repeat_start_index}")

        processed_count = 0
        skipped_count = 0
        corrupted_count = 0
        duplicated_count = 0
        repeated_count = 0
        black_count = 0

        # Store the repeat source image path for camera repeat modes
        repeat_source_filename = None
        if self.fault_mode in ["cam0_repeat", "cam1_repeat"]:
            repeat_source_filename = selected_filenames[self.repeat_source_index - self.start_index]

        # Process each subfolder
        for subfolder_name, images in subfolder_images.items():
            print(f"\nProcessing subfolder: {subfolder_name}")
            
            # Find images in this subfolder that match our selected filenames
            subfolder_selected_images = [img for img in images if img.name in selected_filenames]
            subfolder_selected_images.sort(key=lambda x: selected_filenames.index(x.name))
            
            for relative_path in subfolder_selected_images:
                # Get the original index of this filename
                original_idx = self.start_index + selected_filenames.index(relative_path.name)
                
                src_path = Path(self.original_images_dir) / relative_path
                
                # Ensure output subdirectory exists
                self.ensure_output_dir_structure(relative_path)
                dst_path = Path(self.output_dir) / relative_path

                # Handle cam0_repeat fault mode
                if self.fault_mode == "cam0_repeat":
                    if subfolder_name == "cam0":
                        if original_idx < self.repeat_start_index:
                            # Before repeat start: copy normal image
                            shutil.copy(src_path, dst_path)
                            print(f"  cam0 (normal): {relative_path.name}")
                        else:
                            # After repeat start: copy the repeated image
                            repeat_src_path = Path(self.original_images_dir) / f"cam0/{repeat_source_filename}"
                            if repeat_src_path.exists():
                                shutil.copy(repeat_src_path, dst_path)
                                print(f"  cam0 (REPEAT): {relative_path.name} -> copying {repeat_source_filename}")
                                repeated_count += 1
                            else:
                                print(f"  cam0 (ERROR): Repeat source {repeat_source_filename} not found, copying original")
                                shutil.copy(src_path, dst_path)
                    
                    elif subfolder_name == "cam1":
                        # cam1 always gets correct images
                        shutil.copy(src_path, dst_path)
                        print(f"  cam1 (normal): {relative_path.name}")
                    
                    else:
                        # Other cameras get normal treatment
                        shutil.copy(src_path, dst_path)
                        print(f"  {subfolder_name} (normal): {relative_path.name}")

                # Handle cam1_repeat fault mode
                elif self.fault_mode == "cam1_repeat":
                    if subfolder_name == "cam0":
                        # cam0 always gets correct images
                        shutil.copy(src_path, dst_path)
                        print(f"  cam0 (normal): {relative_path.name}")
                    
                    elif subfolder_name == "cam1":
                        if original_idx < self.repeat_start_index:
                            # Before repeat start: copy normal image
                            shutil.copy(src_path, dst_path)
                            print(f"  cam1 (normal): {relative_path.name}")
                        else:
                            # After repeat start: copy the repeated image
                            repeat_src_path = Path(self.original_images_dir) / f"cam1/{repeat_source_filename}"
                            if repeat_src_path.exists():
                                shutil.copy(repeat_src_path, dst_path)
                                print(f"  cam1 (REPEAT): {relative_path.name} -> copying {repeat_source_filename}")
                                repeated_count += 1
                            else:
                                print(f"  cam1 (ERROR): Repeat source {repeat_source_filename} not found, copying original")
                                shutil.copy(src_path, dst_path)
                    
                    else:
                        # Other cameras get normal treatment
                        shutil.copy(src_path, dst_path)
                        print(f"  {subfolder_name} (normal): {relative_path.name}")

                # Handle cam0_repeat_black fault mode
                elif self.fault_mode == "cam0_repeat_black":
                    if subfolder_name == "cam0":
                        if original_idx < self.repeat_start_index:
                            # Before repeat start: copy normal image
                            shutil.copy(src_path, dst_path)
                            print(f"  cam0 (normal): {relative_path.name}")
                        else:
                            # After repeat start: create black image
                            if self.create_black_image(src_path, dst_path):
                                print(f"  cam0 (BLACK): {relative_path.name} -> black image")
                                black_count += 1
                            else:
                                print(f"  cam0 (ERROR): Could not create black image, copying original")
                                shutil.copy(src_path, dst_path)
                    
                    elif subfolder_name == "cam1":
                        # cam1 always gets correct images
                        shutil.copy(src_path, dst_path)
                        print(f"  cam1 (normal): {relative_path.name}")
                    
                    else:
                        # Other cameras get normal treatment
                        shutil.copy(src_path, dst_path)
                        print(f"  {subfolder_name} (normal): {relative_path.name}")

                # Handle cam1_repeat_black fault mode
                elif self.fault_mode == "cam1_repeat_black":
                    if subfolder_name == "cam0":
                        # cam0 always gets correct images
                        shutil.copy(src_path, dst_path)
                        print(f"  cam0 (normal): {relative_path.name}")
                    
                    elif subfolder_name == "cam1":
                        if original_idx < self.repeat_start_index:
                            # Before repeat start: copy normal image
                            shutil.copy(src_path, dst_path)
                            print(f"  cam1 (normal): {relative_path.name}")
                        else:
                            # After repeat start: create black image
                            if self.create_black_image(src_path, dst_path):
                                print(f"  cam1 (BLACK): {relative_path.name} -> black image")
                                black_count += 1
                            else:
                                print(f"  cam1 (ERROR): Could not create black image, copying original")
                                shutil.copy(src_path, dst_path)
                    
                    else:
                        # Other cameras get normal treatment
                        shutil.copy(src_path, dst_path)
                        print(f"  {subfolder_name} (normal): {relative_path.name}")

                # Handle other fault modes
                elif self.fault_mode == "missing" and original_idx % 10 == 0:
                    print(f"  Skipping (missing): {relative_path}")
                    skipped_count += 1
                    continue

                elif self.fault_mode == "corrupt" and original_idx % 15 == 0:
                    print(f"  Corrupting: {relative_path}")
                    with open(dst_path, 'w') as f:
                        f.write("corrupted data")
                    corrupted_count += 1

                elif self.fault_mode == "duplicate" and original_idx % 20 == 0:
                    print(f"  Duplicating: {relative_path}")
                    shutil.copy(src_path, dst_path)
                    dup_path = dst_path.with_name(dst_path.stem + '_dup' + dst_path.suffix)
                    shutil.copy(src_path, dup_path)
                    duplicated_count += 1

                elif self.fault_mode == "normal":
                    shutil.copy(src_path, dst_path)

                else:
                    # Default to normal copy
                    shutil.copy(src_path, dst_path)

                processed_count += 1

        # Print summary
        print("\n" + "="*60)
        print("DATA GENERATION SUMMARY")
        print("="*60)
        print(f"Total unique timestamps available: {len(unique_filenames)}")
        print(f"Selected range: {self.start_index} to {self.end_index-1}")
        print(f"Images processed: {processed_count}")
        print(f"Fault mode: {self.fault_mode}")
        
        if self.fault_mode in ["cam0_repeat", "cam1_repeat"]:
            camera_name = "Cam0" if self.fault_mode == "cam0_repeat" else "Cam1"
            print(f"{camera_name} images repeated: {repeated_count}")
            print(f"Repeated image source: {repeat_source_filename}")
        elif self.fault_mode in ["cam0_repeat_black", "cam1_repeat_black"]:
            camera_name = "Cam0" if self.fault_mode == "cam0_repeat_black" else "Cam1"
            print(f"{camera_name} black images created: {black_count}")
        elif self.fault_mode == "missing":
            print(f"Images skipped (missing): {skipped_count}")
        elif self.fault_mode == "corrupt":
            print(f"Images corrupted: {corrupted_count}")
        elif self.fault_mode == "duplicate":
            print(f"Images duplicated: {duplicated_count}")
        
        print(f"Output directory: {self.output_dir}")
        
        # Show final subfolder structure
        print("\nOutput subfolder structure:")
        for subfolder_name in subfolder_images.keys():
            output_subfolder = Path(self.output_dir) / subfolder_name
            if output_subfolder.exists():
                count = len([f for f in output_subfolder.iterdir() if f.is_file()])
                print(f"  {subfolder_name}: {count} files")
        
        print("="*60)

# Create and run the data generator
if __name__ == "__main__":
    # Validate input parameters
    if start_index < 0:
        print("Error: start_index must be >= 0")
        exit(1)
    
    if end_index != -1 and end_index <= start_index:
        print("Error: end_index must be greater than start_index or -1 (for all images)")
        exit(1)
    
    generator = GenerateData(
        original_images_dir, 
        output_dir, 
        fault_mode, 
        start_index, 
        end_index,
        repeat_start_index,
        repeat_source_index
    )
    generator.generate()