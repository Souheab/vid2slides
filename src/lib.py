import cv2
from pathlib import Path
from skimage.metrics import structural_similarity
from tqdm import tqdm
from fpdf import FPDF
from PIL import Image
import imagehash
import os

def split_video(video_path, output_folder=Path("./vid2slides_output/")):
    """Splits a video into frames and saves them in a folder.
    
    Args:
        video_path (Path): Path to the video to be split.
        output_folder_path (Path): Path to the folder where the frames will be saved.
    """
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    if not (output_folder / "FILES IN THIS FOLDER WILL BE DELETED.txt").exists():
      with open(output_folder / "FILES IN THIS FOLDER WILL BE DELETED.txt", "w") as f:
        f.write("FILES IN THIS FOLDER WILL BE DELETED SO DO NOT PUT ANYTHING IMPORTANT IN THIS FOLDER\n")

    for file in os.listdir(output_folder):
      os.remove(output_folder / file)
    
    vidcap = cv2.VideoCapture(str(video_path))

    original_fps = vidcap.get(cv2.CAP_PROP_FPS)
    target_fps = 0.25
    frame_skip = max(1, round(original_fps / target_fps)) if original_fps > target_fps else 1
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame = 0
    extracted_frame_count = 0

    chosen_image = None

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
    vidcap.release()

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

def images_to_pdf(filename, image_folder=Path("./vid2slides_output/")):
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

