#!/bin/env python3

from lib import split_video, images_to_pdf
import click
from pathlib import Path

@click.command
@click.argument('filepath', type =click.Path(exists=True))

def run(filepath):
  if filepath is not None:
    split_video(filepath)
    filename = Path(filepath).stem
    images_to_pdf(filename)
    print(f"Created {filename}.pdf")


if __name__ == '__main__':
  run()
