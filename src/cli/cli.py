from datetime import date, datetime

from tqdm.std import tqdm  # type:ignore

from src.adapter import Adapter
from src.app import App
from src.cli.save import Save
from src.constants import VALID_EXTS
from src.logger import Logger


class CLI:
    f"""
    valid file extensions: {VALID_EXTS}
    """
    logger: Logger

    def __init__(self: "CLI") -> None:
        self.logger = Logger()
        self.logger.log(
            """
               ____                          ____                 
              / ____ _____ ___ ___________  / / ___ ____ ____ ____
             _\ \/ // / _ / -_/ __/ __/ _ \/ / / _ `/ _ `/ -_/ __/
            /___/\_,_/ .__\__/_/  \__/\___/_/_/\_,_/\_, /\__/_/   
                    /_/                            /___/          
            """  # noqa
        )
        return

    def collage(
        self: "CLI",
        *inputs: str,
        fname: str = f"{datetime.now()}".replace(" ", "_"),
        dir: str = f"{date.today()}",
        **kwargs,
    ):
        """
        Make a collage:
        ---
        Inputs:
            an image or images via url(s), filepath(s) or directory(s)
        Flags:
            --dir: a directory to save to
            --fname: a file name to save as
        """
        save = Save(fname=fname, dir=dir)
        imgs = list(Adapter.load(*inputs))
        self.logger.log(f"loaded {len(imgs)} images")
        self.logger.log("collaging images:")
        (img,) = App.collage(tqdm(imgs), **kwargs)
        save.jpg(img)
        self.logger.log(f"saved to {dir}/{fname}.jpg")
        return

    def segment(
        self: "CLI",
        *inputs: str,
        fname: str = f"{datetime.now()}".replace(" ", "_"),
        dir: str = f"{date.today()}",
        **kwargs,
    ):
        """
        Cuts out segments:
        ---
        Inputs:
            an image or images via url(s), filepath(s) or directory(s)
        Flags:
            --dir: a directory to save to
            --fname: a file name to save as
        """
        save = Save(fname=fname, dir=dir)
        imgs = list(Adapter.load(*inputs))
        self.logger.log(f"loaded {len(imgs)} images")
        self.logger.log("segmenting images:")
        segments = App.segment(tqdm(imgs), **kwargs)
        for idx, img in enumerate(segments):
            save.png(img, index=idx)
        return

    def masks(
        self: "CLI",
        *inputs: str,
        fname: str = f"{datetime.now()}".replace(" ", "_"),
        dir: str = f"{date.today()}",
    ):
        """
        Generates alpha masks of segments:
        ---
        Inputs:
            an image or images via url(s), filepath(s) or directory(s)
        Flags:
            --dir: a directory to save to
            --fname: a file name to save as
        """
        save = Save(fname=fname, dir=dir)
        imgs = list(Adapter.load(*inputs))
        self.logger.log(f"loaded {len(imgs)} images")
        self.logger.log("segmenting images:")
        masks = App.masks(tqdm(imgs))
        for idx, img in enumerate(masks):
            save.png(img, index=idx)
        return

    def alpha_matte(
        self: "CLI",
        input: str,
        fname: str = f"{datetime.now()}".replace(" ", "_"),
        dir: str = f"{date.today()}",
        **kwargs,
    ):
        """
        Produces an alpha matte for objects in video
        ---
        Inputs:
            a video file
        Flags:
            --dir: a directory to save to
            --fname: a file name to save as
        """
        save = Save(fname=fname, dir=dir)
        try:
            video = Adapter.video(input)
        except OSError as e:
            self.logger.error(str(e))
        self.logger.log("segmenting video:")
        save.mp4(App.alpha_matte(video, **kwargs), fps=video.fps)
        video.close()
        return


if __name__ == "__main__":
    from fire import Fire  # type: ignore

    Fire(CLI())
