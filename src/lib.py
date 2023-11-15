import cv2
from pathlib import Path
from skimage.metrics import structural_similarity
from tqdm import tqdm
from fpdf import FPDF
from PIL import Image
import imagehash
import hashlib
import os

def split_video(video_path, processing_frame_rate):
  """Splits a video into frames and saves them in a folder. """

  hasher = hashlib.sha256()

  # Hash function parameters
  with open(video_path, "rb") as f, tqdm(total=os.path.getsize(video_path), desc="Hashing Parameters", unit="B", unit_scale=True) as pbar:
      for chunk in iter(lambda: f.read(4096), b""):
          hasher.update(chunk)
          pbar.update(len(chunk))

  hasher.update(str(processing_frame_rate).encode("utf-8"))
  hash_str = hasher.hexdigest()
  vid_file_name = video_path.name
  hash_substr = hash_str[:6]
  output_folder = Path(".") / f"{vid_file_name}[{hash_substr}]"
  output_folder.mkdir(parents=True, exist_ok=True)


  warning_file_name = "DO NOT MODIFY THIS FOLDER.txt"

  if not (output_folder / warning_file_name).exists():
    with open(output_folder / warning_file_name, "w") as f:
      f.write("FILES IN THIS FOLDER WILL BE DELETED AND SOME FILES ARE USED FOR THIS PROGRAM'S FUNCTIONING. PLEASE DO NOT MODIFY THIS FOLDER OR STORE IMPORTANT FILES HERE.\n")

  # Check if a file named metadata exists and contains the same hash
  metadata_file = output_folder / "metadata"
  if metadata_file.exists():
    with open(metadata_file, "r") as f:
      metadata_hash = f.read()
      if metadata_hash == hash_str:
          print("Found cached image output with same parameter hash. Skipping video processing.")
          return output_folder

  for file in os.listdir(output_folder):
    os.remove(output_folder / file)

  vidcap = cv2.VideoCapture(str(video_path))

  original_fps = vidcap.get(cv2.CAP_PROP_FPS)
  frame_skip = max(1, round(original_fps / processing_frame_rate)) if original_fps > processing_frame_rate else 1
  total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

  current_frame = 0
  extracted_frame_count = 0

  chosen_image = None

  # Process the video
  with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
    while current_frame < total_frames:
      # Set the position of the next frame to read
      vidcap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

      # Read the frame
      success, image = vidcap.read()
      if not success:
        break

      if chosen_image is None:
        chosen_image = image
      else:
        if are_slides_same(chosen_image, image) and current_frame < total_frames:
          if is_better_image(image, chosen_image):
            chosen_image = image
          else:
            # Save the chosen image
            cv2.imwrite(str(output_folder / f"frame_{extracted_frame_count}.png"), chosen_image)
            extracted_frame_count += 1
            chosen_image = image


      current_frame += frame_skip
      pbar.update(frame_skip)

  # Save the last image
  cv2.imwrite(str(output_folder / f"frame_{extracted_frame_count}.png"), chosen_image)

  # Save the metadata
  with open(metadata_file, "w") as f:
    f.write(hash_str)

  vidcap.release()

  return output_folder

def are_slides_same(image1, image2, similarity_threshold=0.90):
  """Returns True if the two images are similar enough to be considered the same slide. Otherwise, returns False."""
  image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

  image1_gray = cv2.resize(image1_gray, (image2_gray.shape[1], image2_gray.shape[0]))

  # Compute SSIM between the two images
  score, _ = structural_similarity(image1_gray, image2_gray, full=True)

  return score >= similarity_threshold

def is_better_image(image1, image2):
  """If image 1 is better than image 2, return True. Otherwise, return False. Assumes Images are similar"""
  # Check if image 1 has more contrast than image 2
  
  image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  image1_gray = cv2.resize(image1_gray, (image2_gray.shape[1], image2_gray.shape[0]))
  image1_contrast = cv2.Laplacian(image1_gray, cv2.CV_64F).var()
  image2_contrast = cv2.Laplacian(image2_gray, cv2.CV_64F).var()
  return image1_contrast > image2_contrast

def images_to_pdf(filename, image_folder):
  # Open image to get its size
  with Image.open(image_folder / "frame_0.png") as img:
    width, height = img.size

  pdf = FPDF(unit="pt", format=(width, height))

  files = sorted(os.listdir(image_folder))
  for file in files:
    if file.endswith(".png"):
      image_path = os.path.join(image_folder, file)
      pdf.add_page()
      pdf.image(image_path, x=0, y=0, w=width, h=height)

  output_pdf_path = Path(".") / f"{filename}.pdf"
  pdf.output(output_pdf_path)

