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

    def addError(self, test, err):
        super().addError(test, err)
        # Preserve the first failure even if a later concurrency regression
        # triggers the process watchdog before unittest's final summary.
        self.stream.writeln(self._exc_info_to_string(err, test))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.stream.writeln(self._exc_info_to_string(err, test))


if __name__ == "__main__":
    faulthandler.enable()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"))
    runner = unittest.TextTestRunner(verbosity=2, resultclass=WatchedResult)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
