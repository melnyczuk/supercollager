from fire import Fire  # type:ignore

from src.app import App
from src.cli.save import Save
from src.constants import VALID_EXTS


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
        """
        )

    @staticmethod
    def collage(
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
        img = App.collage(list(inputs), **kwargs)
        save.one(img)

    @staticmethod
    def segment(
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
        segments = App.segment(list(inputs), **kwargs)
        save.many(segments)


if __name__ == "__main__":
    Fire(CLI())
