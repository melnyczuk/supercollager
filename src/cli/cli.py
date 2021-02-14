from fire import Fire  # type:ignore

from src.adapters.load import Load
from src.app import App
from src.cli.save import Save
from src.constants import VALID_EXTS
from src.logger import Logger


class CLI:
    f"""

    valid file extensions: {VALID_EXTS}
    """

    def __init__(self):
        print(
            """
               ____                          ____                 
              / ____ _____ ___ ___________  / / ___ ____ ____ ____
             _\ \/ // / _ / -_/ __/ __/ _ \/ / / _ `/ _ `/ -_/ __/
            /___/\_,_/ .__\__/_/  \__/\___/_/_/\_,_/\_, /\__/_/   
                    /_/                            /___/          
        """  # noqa
        )
        self.logger = Logger()

    def collage(
        self: "CLI",
        *inputs: str,
        fname: str = None,
        dir: str = None,
        **kwargs,
    ):
        """
        Make a collage:
        ---
        Inputs:
            an image or images via url(s), filepath(s) or directory(s)
        Flags:
            --dir: a directory to save to    (required)
            --fname: a file name to save as  (required)
        """
        save = Save(fname=fname, dir=dir)
        imgs = Load.uris(inputs)
        self.logger.log(f"loaded {len(imgs)} images")
        self.logger.log("collaging images:")
        img = App.collage(imgs, **kwargs)
        save.one(img)

    @staticmethod
    def segment(
        self: "CLI",
        *inputs: str,
        fname: str = None,
        dir: str = None,
        **kwargs,
    ):
        """
        Cuts out segments:
        ---
        Inputs:
            an image or images via url(s), filepath(s) or directory(s)
        Flags:
            --dir: a directory to save to    (required)
            --fname: a file name to save as  (required)
        """
        save = Save(fname=fname, dir=dir)
        imgs = Load.uris(inputs)
        self.logger.log(f"loaded {len(imgs)} images")
        self.logger.log("segmenting images:")
        segments = App.segment(imgs, **kwargs)
        self.logger.log(f"found {len(segments)} segments in {len(imgs)} URIs")
        save.many(segments)


if __name__ == "__main__":
    Fire(CLI())
