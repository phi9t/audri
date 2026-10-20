"""Container test entry point with a watchdog for concurrency regressions."""

import faulthandler
from pathlib import Path
import unittest


class WatchedResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        # A hung test must produce diagnostic stacks, not consume a CI runner
        # indefinitely. This suite uses tiny fixtures and no real model loads.
        faulthandler.dump_traceback_later(45, exit=True)

    def stopTest(self, test):
        faulthandler.cancel_dump_traceback_later()
        super().stopTest(test)


if __name__ == "__main__":
    faulthandler.enable()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"))
    runner = unittest.TextTestRunner(verbosity=2, resultclass=WatchedResult)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
