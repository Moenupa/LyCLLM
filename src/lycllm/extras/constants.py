import os

AUDIO_PLACEHOLDER = os.getenv("AUDIO_PLACEHOLDER", "<audio>")
IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")
VIDEO_PLACEHOLDER = os.getenv("VIDEO_PLACEHOLDER", "<video>")
IGNORE_INDEX = -100


def get_seed() -> int | None:
    """Return the Lightning seed from the environment, if available.

    Returns:
        int | None: Integer seed parsed from the ``PL_GLOBAL_SEED`` environment
            variable, or ``None`` if the variable is not set.
    """
    if seed := os.getenv("PL_GLOBAL_SEED"):
        return int(seed)

    return None
