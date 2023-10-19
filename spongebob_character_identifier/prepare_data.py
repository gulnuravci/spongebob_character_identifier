import os

from PIL import Image

def convert_to_jpg(input_folder):
  """Converts the images in input folder to .jpg"""
  # Check if input folder exists
  if os.path.exists(input_folder):
      print("here")
      # Get the name of the input folder
      folder_name = os.path.basename(os.path.normpath(input_folder))

      # Create the output folder if it doesn't exist
      output_folder = f'{folder_name}_jpg'
      if not os.path.exists(output_folder):
          os.makedirs(output_folder)

      # List all files in the input folder
      files = os.listdir(input_folder)

      for idx, filename in enumerate(files):
          input_path = os.path.join(input_folder, filename)
          # Create a new filename based on folder name and index
          output_filename = f'{folder_name}_{idx}.jpg'
          output_path = os.path.join(output_folder, output_filename)

          try:
              # Open and convert the image to JPG format
              with Image.open(input_path) as img:
                  img.convert('RGB').save(output_path, 'JPEG')
              print(f'Converted {input_path} to {output_path}')
          except Exception as e:
              print(f'Error converting {input_path}: {str(e)}')
  else:
      print("The input folder doesn't exist")