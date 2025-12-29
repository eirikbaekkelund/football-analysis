import os
from SoccerNet.Downloader import SoccerNetDownloader


def download_tracking_data(local_dir: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=local_dir)
    downloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])
    downloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])


def download_pitch_calibration(local_dir: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=local_dir)
    downloader.downloadDataTask(task="calibration", split=["train", "test", "challenge"])


if __name__ == "__main__":
    # LOCAL_DIR = "soccernet/tracking/"
    # download_tracking_data(LOCAL_DIR)
    LOCAL_DIR = "soccernet/calibration/"
    download_pitch_calibration(LOCAL_DIR)
