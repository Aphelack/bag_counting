from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Path("./storage")

    # RTMDet checkpoint, produced by ../training/ (see its README). Left
    # unset by default so the app still runs — with StubDetector standing
    # in, always reporting 0 bags — on machines without a GPU/checkpoint,
    # e.g. this project's own dev environment. (Named detector_*, not
    # model_*, to avoid Pydantic v2's reserved `model_` field-name warning.)
    detector_config_path: Path | None = None
    detector_checkpoint_path: Path | None = None
    detector_device: str = "cuda:0"
    detection_score_threshold: float = 0.3

    @property
    def input_dir(self) -> Path:
        return self.data_dir / "input"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"


settings = Settings()
settings.input_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
