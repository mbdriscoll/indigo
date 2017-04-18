import random
import logging

log = logging.getLogger(__name__)


class Oracle(object):
    def __init__(self):
        self.num_questions = 0

    def ask(self, question):
        self.num_questions += 1
        return self._ask()

    def _ask(self, question):
        raise NotImplementedError()


class RandomOracle(Oracle):
    def _ask(self):
        return random.choice((True, False))


class YesOracle(Oracle):
    def _ask(self):
        return True


class NoOracle(Oracle):
    def _ask(self):
        return False


class MinimizeMemoryUsage(Oracle):
    pass

class MinimizeMemoryBandwidth(Oracle):
    pass

class MinimizeFlops(Oracle):
    pass

class PredeterminedSchedule(Oracle):
    pass
