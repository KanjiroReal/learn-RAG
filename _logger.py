import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    backtrace=True, 
    diagnose=True,
    level="DEBUG"
)