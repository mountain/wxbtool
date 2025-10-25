
def ci_short_circuit(func):
    """Decorator to short-circuit training, validation, or testing in CI environments."""

    def wrapper(self, *args, **kwargs):
        batch_idx = args[1]  # Assuming the second argument is batch_idx
        if self.ci:
            # Short-circuit logic for CI environment
            if func.__name__ == "training_step" and batch_idx > 0:
                return
            elif func.__name__ == "validation_step" and batch_idx > 1:
                return
            elif func.__name__ == "test_step" and batch_idx > 0:
                return
        return func(self, *args, **kwargs)

    return wrapper


def ci_batch_injection(batch_size_ci=2):
    """Decorator to adjust batch size for CI environments."""

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if self.ci:
                self.opt.batch_size = batch_size_ci
            return func(self, *args, **kwargs)

        return wrapper

    return decorator