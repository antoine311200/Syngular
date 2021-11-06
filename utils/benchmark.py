import time
import threading

import tensorflow as tf


""" Google implementation from its tensorflow tutorial 'NMT with Attention' """
class ShapeChecker():
    def __init__(self):
        # Keep a cache of every axis-name seen
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return

        if isinstance(names, str):
            names = (names,)

        shape = tf.shape(tensor)
        rank = tf.rank(tensor)

        if rank != len(names):
            raise ValueError(f'Rank mismatch:\n'
                        f'    found {rank}: {shape.numpy()}\n'
                        f'    expected {len(names)}: {names}\n')

        for i, name in enumerate(names):
            if isinstance(name, int):
                old_dim = name
            else:
                old_dim = self.shapes.get(name, None)
            new_dim = shape[i]

            if (broadcast and new_dim == 1):
                continue

            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                f"    found: {new_dim}\n"
                                f"    expected: {old_dim}\n")

class Timer:

    @classmethod
    def wrapper(self, func):
        def wrap(*args, **kwargs):
            name = func.__qualname__

            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            cost = end - start

            print(f'`{name}` time : {cost}')
            
            return result
        return wrap



class Thread:

    def __init__(self) -> None:
        self.event = threading.Event()
        self.thread = threading.Thread()
        self.stop = False
    
    def interval(self, action, elapse, kill=10000):
        self.thread = threading.Thread(target=self._interval, args=(action, elapse, lambda: self.stop, ))
        self.start()

        self.kill(kill)

        return self

    @Timer.wrapper
    def _interval(self, action, elapse, stop):

        next =  time.time() + elapse

        while not self.event.wait(next-time.time()):
            next += elapse
            action()
            if stop():
                break

    def start(self):
        self.thread.start()
    
    def kill(self, elapse):
        while not self.event.wait(elapse):
            self.event.set()
            self.stop = True


if __name__ == "__main__":

    def action():
        print("Hey")

    # thread = Thread()
    # inter = thread.interval(action, 0.6, kill=2)

