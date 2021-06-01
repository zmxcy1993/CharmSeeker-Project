import os
import sys
import time


def safe_delete(filename):
    cmd = f'mv {filename} {filename}.delete && rm {filename}.delete'
    fail = os.system(cmd)
    return not fail


class Locker:

    def __init__(self):
        self.locks = {}

    def __del__(self):
        for filename in list(self.locks):
            self.locks[filename] = 1
            self.unlock(filename)

    def lock(self, filename):
        if filename in list(self.locks):
            self.locks[filename] += 1
            return True
        else:
            cmd = f'ln -s /dev/null "{filename}.lock" 2> /dev/null'
            fail = os.system(cmd)
            if not fail:
                self.locks[filename] = 1
            return not fail

    def unlock(self, filename):
        if filename not in list(self.locks):
            # sys.stderr.write(f"Trying to unlock not-locked file {filename}.\n")
            return True
        if self.locks[filename] == 1:
            success = safe_delete(f"{filename}.lock")
            if not success:
                sys.stderr.write(f"Could not unlock file: {filename}.\n")
            del self.locks[filename]
            return success
        else:
            self.locks[filename] -= 1
            return True

    def lock_wait(self, filename):
        while not self.lock(filename):
            time.sleep(0.01)

