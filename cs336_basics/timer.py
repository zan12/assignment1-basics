import time

class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self,):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        # print(f"{self.name} Time Elapsed: {self.end_time-self.start_time:.4f}")