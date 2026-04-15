import os
from s3fs import S3FileSystem
import subprocess
import pandas as pd


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
    nuts_3: str,
    year: str,
) -> None:
    """
    Download data for a specific context, if not already downloaded.

    Args:
        patchs_path (str): Paths to patchs.
        labels_path (str): Paths to labels.
        nuts_3 (str): NUTS3.
        year (str): Year.
    """
    all_exist = all(os.path.exists(f"{directory}") for directory in [patchs_path, labels_path])

    if all_exist:
        return None

    alias_cmd = [
        "mc", "alias", "set", "public",
        "https://minio.lab.sspcloud.fr",
        "", ""
    ]

    url_filenames = f"https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project3/data/images/{nuts_3}/{year}/filename2bbox.parquet"
    df_filenames = pd.read_parquet(url_filenames)
    filenames_patchs = df_filenames.filename.tolist()
    filenames_labels = [filename.split('.')[0]+'.npy' for filename in filenames_patchs]

    print("Downloading data from S3...\n")
    with open("/dev/null", "w") as devnull:
        # set public alias
        subprocess.run(alias_cmd, check=True, stdout=devnull, stderr=devnull)

        # download patchs
        for filename_patch in filenames_patchs:
            patch_cmd = [
                "mc",
                "cp",
                f"public/projet-formation/diffusion/funathon/2026/project3/data/images/{nuts_3}/{year}/{filename_patch}",  # noqa
                f"data/data-preprocessed/patchs/{nuts_3}/{year}/",
            ]
            subprocess.run(patch_cmd, check=True, stdout=devnull, stderr=devnull)

        # download normalization metrics
        normalization_metrics_cmd = [
            "mc",
            "cp",
            f"public/projet-formation/diffusion/funathon/2026/project3/data/images/{nuts_3}/{year}/metrics-normalization.yaml",  # noqa
            f"data/data-preprocessed/patchs/{nuts_3}/{year}/",
        ]
        subprocess.run(normalization_metrics_cmd, check=True, stdout=devnull, stderr=devnull)

        # download filename2bbox
        filename2bbox_cmd = [
            "mc",
            "cp",
            f"public/projet-formation/diffusion/funathon/2026/project3/data/images/{nuts_3}/{year}/filename2bbox.parquet",  # noqa
            f"data/data-preprocessed/patchs/{nuts_3}/{year}/",
        ]
        subprocess.run(filename2bbox_cmd, check=True, stdout=devnull, stderr=devnull)

        # download labels
        for filename_label in filenames_labels:
            label_cmd = [
                "mc",
                "cp",
                f"public/projet-formation/diffusion/funathon/2026/project3/data/images/{nuts_3}/{year}/{filename_label}",  # noqa
                f"data/data-preprocessed/labels/{nuts_3}/{year}/",
            ]
            subprocess.run(label_cmd, check=True, stdout=devnull, stderr=devnull)
    print("Downloading finished!\n")