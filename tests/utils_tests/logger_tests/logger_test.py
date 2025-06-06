from FRsutils.utils.logger.logger_util import get_logger, _TinyLogger
# import logging

# logger = _TinyLogger(
#     log_to_console=False,
#     log_to_file=True,
#     log_file_extension='csv',  # or "json" or None
#     file_path="log_output.json",
#     level=logging.DEBUG
# )
logger = get_logger()

logger.attach_exception_hook()


def crash():
    return 1 / 0  # Uncaught ZeroDivisionError

def evaluate_model2():

    try:
        1 / 0
    except ZeroDivisionError:
        
        logger.error("Division by zero in evaluation.")
        logger.critical("Critical error in evaluation.")
 

def evaluate_model():
    logger.info("Evaluating model...")
    logger.warning("Validation data is imbalanced.")


evaluate_model()
# for i in range(100):
#     evaluate_model()
evaluate_model2()



# crash()
