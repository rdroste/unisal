"""
Script used to export DHF1K from original form (after downloading + unar files) to a directory
compatible with this repository
"""
from pathlib import Path
from shutil import copytree

import cv2

# path to https://drive.google.com/drive/folders/1sW0tf9RQMO4RR7SyKhU8Kmbm4jwkFGpQ
# DHF1K_PATH = "/Path/To/Dataset/DHF1K"
DHF1K_PATH = "/Users/Alessandro/Datasets/DHF1K"


def extract_frames(video_path, frames_dir):
    vidcap = cv2.VideoCapture(str(video_path))
    success, image = vidcap.read()
    count = 1
    while success:
        frame_filename = f"{count:04d}.png"
        cv2.imwrite(str(frames_dir / frame_filename), image)  # Save frame as PNG file
        success, image = vidcap.read()
        count += 1
    vidcap.release()


def convert_dhf1k_to_unisal_format(dhf1k_path: str):
    dhf1k_path = Path(dhf1k_path)
    dhf1k_unisal_path = Path(__file__).parent.parent / "data" / "DHF1K"
    dhf1k_unisal_path.mkdir(parents=True, exist_ok=True)

    dhf1k_path_annot = dhf1k_path / "annotation"
    # only for debugging purposes
    if (dhf1k_unisal_path / "annotation").exists() is False:
        copytree(dhf1k_path_annot, dhf1k_unisal_path / "annotation")

    for idx, video in enumerate(sorted((dhf1k_path / "video").glob("*.AVI"))):
        # original videos in DHF1K are named like 001.AVI, OO2.AVI
        # but the annotations are in the format like 0001, 0002, 0003.
        # Hence we need to manually add a "0" to the video naming to
        # check if the folder already exists within the "annotations"
        # subfolder. The reason why we need to do this check is
        # because unisal expects all the images extracted from videos
        # to be present within the "annotation" subfolder. This folder
        # that has been copied from the original DHF1K structure however
        # contains only 700 folders that correspond to the ones that include
        # label, the remaining 300 folder used for testing are inserted thanks
        # to this check

        # naming is like 0001, 0999 ... 1000 ... hence for the last folder we
        # do not add the extra 0
        annot_vs_video = 0 if idx != 999 else ""
        video_folder_name = f'{str(annot_vs_video)}{video.parts[-1].replace(".AVI", "")}'
        video_subfolder = Path(dhf1k_unisal_path, "annotation", video_folder_name, "images")
        
        # creates folders form 701 to 1000 and "images" subfolders
        if video_subfolder.exists() is False:
            video_subfolder.mkdir(parents=True, exist_ok=True)

        extract_frames(video_path=video, frames_dir=video_subfolder)


def main():
    convert_dhf1k_to_unisal_format(DHF1K_PATH)


if __name__ == "__main__":
    main()
