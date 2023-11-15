import cv2
from pathlib import Path
from skimage.metrics import structural_similarity
from tqdm import tqdm
from fpdf import FPDF
from PIL import Image
import imagehash
import hashlib
import os

def create_warning_file(folder):
    warning_file_name = "DO NOT MODIFY THIS FOLDER.txt"
    if not (folder / warning_file_name).exists():
        with open(folder / warning_file_name, "w") as f:
            f.write("FILES IN THIS FOLDER WILL BE DELETED AND SOME FILES ARE USED FOR THIS PROGRAM'S FUNCTIONING. PLEASE DO NOT MODIFY THIS FOLDER OR STORE IMPORTANT FILES HERE.\n")

def create_main_cache_directory():
    current_os_type = os.name
    main_cache_dir = None

    if current_os_type == "posix":
        main_cache_dir = Path(os.environ.get("XDG_CACHE_HOME")) / "vid2slides"
        if main_cache_dir is None:
            xdg_cache_dir = os.path.expanduser("~/.cache")
            main_cache_dir = os.path.join(xdg_cache_dir, "vid2slides")

        main_cache_dir.mkdir(parents=True, exist_ok=True)
        create_warning_file(main_cache_dir)


    return main_cache_dir

def get_parameters_hash(video_path, processing_frame_rate):
    hasher = hashlib.sha256()

    with open(video_path, "rb") as f, tqdm(total=os.path.getsize(video_path), desc="Hashing Parameters", unit="B", unit_scale=True) as pbar:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
            pbar.update(len(chunk))

    hasher.update(str(processing_frame_rate).encode("utf-8"))
    hash_str = hasher.hexdigest()
    return hash_str

def get_cache_dir_otherwise_set_output_dir(hash_str, vid_file_name):
    main_cache_dir = create_main_cache_directory()
    hash_substr = hash_str[:6]
    folder_format = f"{vid_file_name}[{hash_substr}]"

    directory_list = [Path(".") / folder_format]
    if main_cache_dir is not None:
        directory_list.insert(0, main_cache_dir / folder_format)

    for directory in directory_list:
        metadata_file = directory / "metadata"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata_hash = f.read()
                if metadata_hash == hash_str:
                    return directory, True


    if main_cache_dir is not None:
        output_dir = main_cache_dir / folder_format
        output_dir.mkdir(parents=True, exist_ok=True)

    else:
        output_dir = Path(".") / folder_format
        output_dir.mkdir(parents=True, exist_ok=True)
        
    for file in os.listdir(output_dir):
        os.remove(output_dir / file)

    return output_dir, False

def split_video(video_path, processing_frame_rate):
    """Splits a video into frames and saves them in a folder. """

    vid_file_name = video_path.name
    hash_str = get_parameters_hash(video_path, processing_frame_rate)
    hash_substr = hash_str[:6]

    output_dir, cache_dir_exists_p = get_cache_dir_otherwise_set_output_dir(hash_str, vid_file_name)
    if cache_dir_exists_p:
        print("Using cached frames, skipping processing.")
        return output_dir
        
    create_warning_file(output_dir)


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
                    cv2.imwrite(str(output_dir / f"frame_{extracted_frame_count}.png"), chosen_image)
                    extracted_frame_count += 1
                    chosen_image = image


            current_frame += frame_skip
            pbar.update(frame_skip)

    # Save the last image
    cv2.imwrite(str(output_dir / f"frame_{extracted_frame_count}.png"), chosen_image)

    # Save the metadata
    metadata_file = output_dir / "metadata"
    with open(metadata_file, "w") as f:
        f.write(hash_str)

    vidcap.release()

    return output_dir

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

