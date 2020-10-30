import numpy as np


def np_to_png(
    arr: np.ndarray,
    fname: str = None,
    dir: str = None,
) -> None:
    from os import path
    import png  # type: ignore

    if not dir:
        raise ValueError("pls provide dir to save to")
    if not fname:
        raise ValueError("pls provide file name to save as")

    out_path = path.join(dir, fname.split(".png")[0])
    png.from_array(arr, mode="L").save(f"{out_path}.png")
