#!/bin/env python3

from lib import split_video, images_to_pdf
import click
from pathlib import Path
import shutil

@click.command
@click.argument('filepath', type =click.Path(exists=True))
@click.option('--processing-fps', default=0.2, help='Frame rate at which video is processed (Lower is faster)')
@click.option('--delete-output-folder', is_flag=True, help='Delete the output folder after processing')

def run(filepath, processing_fps, delete_output_folder):
    if filepath is not None:
        output_images_folder = split_video(Path(filepath), processing_fps)
        filename = Path(filepath).stem
        images_to_pdf(filename, output_images_folder)
        print(f"Created {filename}.pdf")
        if delete_output_folder:
            shutil.rmtree(output_images_folder)
            print(f"Deleted output folder: {output_images_folder}")

if __name__ == '__main__':
    run()
