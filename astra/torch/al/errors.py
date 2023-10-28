class AcquisitionMismatchError(Exception):
    def __init__(self, required, actual):
        super().__init__(f"Acquisition must be an instance of {required}, got {actual}")
