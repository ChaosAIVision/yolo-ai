"""BentoML service entry point (safe GPU-1, single worker)."""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

from src.config import (
    BENTO_SERVICE_NAME,
    logger,
)
from src.deploy.bentoml_service import create_bentoml_service

logger = logging.getLogger(__name__)

# Find model weights
model_path = os.getenv("MODEL_WEIGHTS_PATH")

# if not os.path.exists(model_path):
#     raise FileNotFoundError(
#         f"Model weights not found: {model_path}. "
#         f"Please set MODEL_WEIGHTS_PATH environment variable."
#     )

# Legacy style service
svc = create_bentoml_service(model_path)
logger.info(f"BentoML service initialized: {BENTO_SERVICE_NAME}")

