import os
from s3fs import S3FileSystem
import subprocess


def get_file_system() -> S3FileSystem:
    """
    Return the configured S3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=""
    )


def download_data(
    patchs_path: str,
    labels_path: str,
    source: str,
    dep: str,
    year: str,
    tiles_size: str,
    type_labeler: str,
) -> None:
    """
    Download data from S3 using mc if not already present locally.
    """

    all_exist = all(
        os.path.exists(directory)
        for directory in [patchs_path, labels_path]
    )

    if all_exist:
        return

    patch_cmd = [
        "mc", "cp", "-r",
        f"s3/projet-hackathon-ntts-2025/data-preprocessed/patchs/"
        f"{type_labeler}/{source}/{dep}/{year}/{tiles_size}/",
        f"data/data-preprocessed/patchs/{source}/{dep}/{year}/{tiles_size}/",
    ]

    label_cmd = [
        "mc", "cp", "-r",
        f"s3/projet-hackathon-ntts-2025/data-preprocessed/labels/"
        f"{type_labeler}/{source}/{dep}/{year}/{tiles_size}/",
        f"data/data-preprocessed/labels/{type_labeler}/{source}/{dep}/{year}/{tiles_size}/",
    ]

    print("Downloading data from S3...\n")

    with open("/dev/null", "w") as devnull:
        subprocess.run(patch_cmd, check=True, stdout=devnull, stderr=devnull)
        subprocess.run(label_cmd, check=True, stdout=devnull, stderr=devnull)

    print("Downloading finished!\n")
