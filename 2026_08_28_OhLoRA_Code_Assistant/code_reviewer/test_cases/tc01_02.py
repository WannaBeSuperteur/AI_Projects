import logging

logger = logging.getLogger()

logger.warning("server error detected")
logger.error("server fault")
logger.info("info")
logger.debug("debug")
logger.debug('debug test')
logger.critical("CRITICAL ERROR DETECTED !!")

print('test')
print("testing ...")
print(1)
print(logger)
print(2 + len(str(logger)))
