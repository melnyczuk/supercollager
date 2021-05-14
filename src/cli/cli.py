from datetime import date, datetime

from tqdm.std import tqdm  # type:ignore

from src.adapter import Adapter
from src.app import App
from src.cli.save import Save
from src.constants import VALID_EXTS
from src.logger import Logger

DEFAULT_DIR = f"{date.today()}"
DEFAULT_FNAME = f"{datetime.now()}".replace(" ", "_")


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
        fname: str = DEFAULT_FNAME,
        dir: str = DEFAULT_DIR,
        **kwargs,
    ) -> None:
        """
        Make a collage:
        ---
        Inputs:
            an image or images via url(s), filepath(s) or directory(s)
        Flags:
            -background: int|tuple, for background colour
            -color: float, post-process colour amount
            -contrast: float, post-process contrast amount
            -dir: str, directory to save to
            -fname: str, file name to save as
            -rotate: bool|float, either True (90°) or angle in deg
            -shuffe: bool, whether to shuffle input images
        """
        save = Save(fname=fname, dir=dir)
        imgs = Adapter.load(inputs)
        self.logger.log("collaging images:")
        (img,) = App.collage(tqdm(imgs), **kwargs)
        save.jpg(img)
        self.logger.log(f"saved to {dir}/{fname}.jpg")
        return

    def segment(
        self: "CLI",
        *inputs: str,
        fname: str = DEFAULT_FNAME,
        dir: str = DEFAULT_DIR,
        **kwargs,
    ) -> None:
        """
        Cuts out segments:
        ---
        Inputs:
            an image or images via url(s), filepath(s) or directory(s)
        Flags:
            -dir: str, directory to save to
            -fname: str, file name to save as
            -rotate: bool|float, either True (90°) or angle in deg
            -shuffe: bool, whether to shuffle input images
        """
        save = Save(fname=fname, dir=dir)
        imgs = Adapter.load(inputs)
        self.logger.log("segmenting images:")
        segments = App.segment(tqdm(imgs), **kwargs)
        for idx, img in enumerate(segments):
            save.png(img, index=idx)
        self.logger.log(f"saved to {dir}")
        return

    def masks(
        self: "CLI",
        *inputs: str,
        fname: str = DEFAULT_FNAME,
        dir: str = DEFAULT_DIR,
        **kwargs,
    ) -> None:
        """
        Generates alpha masks of segments:
        ---
        Inputs:
            an image or images via url(s), filepath(s) or directory(s)
        Flags:
            -dir: str, directory to save to
            -fname: str, file name to save as
        """
        save = Save(fname=fname, dir=dir)
        imgs = Adapter.load(inputs)
        self.logger.log("segmenting images:")
        masks = App.masks(tqdm(imgs))
        for idx, img in enumerate(masks):
            save.png(img, index=idx)
        self.logger.log(f"saved to {dir}")
        return

    def alpha_matte(
        self: "CLI",
        *inputs: str,
        fname: str = DEFAULT_FNAME,
        dir: str = DEFAULT_DIR,
        **kwargs,
    ) -> None:
        """
        Produces an alpha matte for objects in video
        ---
        Inputs:
            a video file
        Flags:
            -blur: float, blur mask
            -confidence_threshold: float, confidence for segment mask inclusion
            -dir: str, directory to save to
            -fname: str, file name to save as
            -gain: float, boost mask
            -keyframe_interval: int, how often to use a keyframe
        """
        save = Save(fname=fname, dir=dir)
        (inp,) = inputs
        try:
            video = Adapter.video(inp)
        except OSError as e:
            self.logger.error(str(e))
        self.logger.log("segmenting video:")
        save.mp4(App.alpha_matte(video, **kwargs), fps=video.fps)
        video.close()
        return

    def super_resolution(
        self: "CLI",
        *inputs: str,
        fname: str = DEFAULT_FNAME,
        dir: str = DEFAULT_DIR,
        **kwargs,
    ) -> None:
        """
        Produces an alpha matte for objects in video
        ---
        Inputs:
            an image or images via url(s), filepath(s) or directory(s)
        Flags:
            -device: str, torch device for GPU ("cuda") or CPU ("cpu")
            -dir: str, directory to save to
            -fname: str, file name to save as
        """
        save = Save(fname=fname, dir=dir)
        imgs = Adapter.load(inputs)
        self.logger.log("upscaling images")
        sr = App.super_resolution(imgs, **kwargs)
        for i, img in enumerate(sr):
            save.jpg(img, i)
        self.logger.log(f"saved to {dir}")
        return

    def abstract(
        self: "CLI",
        *inputs: str,
        fname: str = DEFAULT_FNAME,
        dir: str = DEFAULT_DIR,
        **kwargs,
    ) -> None:
        """
        Produces an abstract composition
        ---
        Inputs:
            an image or images via url(s), filepath(s) or directory(s)
        Flags:
            -color: float, post-process colour amount
            -contrast: float, post-process contrast amount
            -device: str, torch device for GPU ("cuda") or CPU ("cpu")
            -dir: str, directory to save to
            -dsize: tuple[int, int], target size of output image
            -fname: str, file name to save as
            -limit: int, how many segments to cut from input images
            -n_segments: int, how many segments to use in composition
            -rotate: bool|float, either True (90°) or angle in deg
            -shuffe: bool, whether to shuffle input images (default True)
            -sr_cycles: int, how many times to upscale using ESRGAN
        """
        save = Save(fname=fname, dir=dir)
        imgs = Adapter.load(inputs)
        self.logger.log("making abstract composition")
        (abst,) = App.abstracts(imgs, **kwargs)
        save.jpg(abst)
        self.logger.log(f"saved to {dir}/{fname}.jpg")
        return


if __name__ == "__main__":
    from fire import Fire  # type: ignore

    Fire(CLI())
