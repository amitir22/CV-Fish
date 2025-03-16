from datetime import datetime

class Timer:
    """
    Microseconds timer
    """
    def __init__(self):
        self.start_time = None
    
    def tick(self):
        self.start_time = datetime.now()
    
    def tock(self):
        """
        Returns:
            (tock - tick) in microseconds
        """
        end_time = datetime.now()
        delta = end_time - self.start_time

        total_days = delta.days
        total_seconds = total_days * 24 * 60 * 60 + delta.seconds
        total_microseconds = total_seconds * 1_000_000 + delta.microseconds

        return total_microseconds


def test():
    """
    Unit test simple function to test this module
    """
    timer = Timer()
    timer.tick()
    for _ in range(1000000):
        pass
    duration = timer.tock()
    print(duration)
    duration = timer.tock()
    print(duration)


if __name__ == '__main__':
    test()
