import numpy as np
import torch

from .RRDBNet import RRDBNet

MODEL_PATH = "weights/esrgan/RRDB_ESRGAN_x4.pth"


class ESRGAN:
    device: torch.device
    model: RRDBNet

    def __init__(
        self: "ESRGAN",
        device: str = "cuda",  # or "cpu"
    ) -> None:
        model = RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(MODEL_PATH), strict=True)
        model.eval()

        self.device = torch.device(device)
        self.model = model.to(self.device)
        return

    def run(self: "ESRGAN", img: np.ndarray) -> np.ndarray:
        lr = (
            torch.from_numpy(
                np.transpose(  # type:ignore
                    img[:, :, (2, 1, 0)],  # type:ignore
                    (2, 0, 1),
                )
                * 1.0
                / 255
            )
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            output = (
                self.model(lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            )

        return (
            np.transpose(  # type:ignore
                output[(2, 1, 0), :, :],
                (1, 2, 0),
            )
            * 255.0
        ).astype(np.uint8)
