from src.pipelines import blocks, collage, segment


class CLI:
    blocks = blocks
    collage = collage
    segment = segment


if __name__ == "__main__":
    from fire import Fire  # type:ignore

    Fire(CLI)
