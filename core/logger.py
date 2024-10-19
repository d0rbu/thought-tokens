import sys

from loguru import logger


logger.remove()
logger.add("training_logs.log", format="{time} {level} {message}", level="DEBUG")
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
