import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    backtrace=True, 
    diagnose=True,
    level="INFO" # DEBUG FOR DEBUG; SUCCESS FOR SHORTER LOG
)