import time
import logging


def time_it(func):
    """
    Decorator to measure and log the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Log the execution time
        logging.info(f"Function {func.__name__} completed in {elapsed_time:.4f} seconds.")
        
        return result

    return wrapper

