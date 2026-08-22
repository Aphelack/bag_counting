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
    # Frames per batched inference call. Conservative default for small/CPU
    # setups — raise via env var on bigger GPUs (e.g. 32-64 on an A100).
    detection_batch_size: int = 8

    @property
    def input_dir(self) -> Path:
        return self.data_dir / "input"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


settings = Settings()
settings.input_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.jobs_dir.mkdir(parents=True, exist_ok=True)
