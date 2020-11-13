from datetime import datetime

from . import pipelines
from .app.io import IO


def save(fn):
    @staticmethod
    def f(*args, **kwargs):
        now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        dir = kwargs.pop("dir", "./dump/output")
        fname = kwargs.pop("fname", now)
        ext = kwargs.pop("ext", "png")

        for i, img in enumerate(fn(*args, **kwargs)):
            IO.save.image(
                img,
                fname=f"{fname}-{i}",
                dir=dir,
                ext=ext,
            )

    return f


class CLI:
    blocks = save(pipelines.blocks)
    collage = save(pipelines.collage)
    segment = save(pipelines.segment)


if __name__ == "__main__":
    from fire import Fire  # type:ignore

    Fire(CLI)
