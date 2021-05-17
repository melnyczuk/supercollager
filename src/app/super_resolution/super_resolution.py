class SuperResolution:
    @staticmethod
    def esrgan(device: str = "cuda"):
        from src.app.super_resolution.esrgan import ESRGAN

        return ESRGAN(device=device)
