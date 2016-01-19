
import timeit
import copy

from pyphi import compute, config, constants, examples, Subsystem
from .subsystem import clear_subsystem_caches


def _clear_joblib_cache():
    constants.joblib_memory.clear()


class BenchmarkConstellation:

    params = ['basic', 'rule154', 'fig16']

    def setup(self, network):
        if network == 'basic':
            self.subsys = examples.basic_subsystem()

        elif network == 'rule154':
            network = examples.rule154_network()
            state = (1,) * 5
            self.subsys = Subsystem(network, state, network.node_indices)

        elif network == 'fig16':
            network = examples.fig16()
            state = (0,) * 7
            self.subsys = Subsystem(network, state, network.node_indices)

        else:
            raise

    def time_constellation(self, network):
        clear_subsystem_caches(self.subsys)
        # network purview caches are left intact
        compute.constellation(self.subsys)


class BenchmarkMainComplex():

    params = [
        ['parallel', 'sequential'],
        ['basic', 'rule154', 'fig16']
    ]
    param_names = ['mode', 'network']

    # Use `default_timer` (clock time) instead of process time because
    # parallel execution spawns separate processes which are not counted
    # in process time.
    # Additionally, we don't need to clear any network caches because we
    # only run 1 iteration.
    # TODO: do we need to clear global caches?
    timer = timeit.default_timer
    number = 1
    repeat = 1
    timeout = 10000

    def setup(self, mode, network):

        if network == 'basic':
            self.network = examples.basic_network()
            self.state = (0, 1, 1)
        elif network == 'rule154':
            self.network = examples.rule154_network()
            self.state = (0, 1, 0, 1, 1)
        elif network == 'fig16':
            self.network = examples.fig16()
            self.state = (1, 0, 0, 1, 1, 1, 0)
        else:
            raise

        # Save config
        self.default_config = copy.copy(config.__dict__)

        # Execution mode
        if mode == 'parallel':
            config.PARALLEL_CUT_EVALUATION = True
        elif mode == 'sequential':
            config.PARALLEL_CUT_EVALUATION = False
        else:
            raise

        config.CACHE_BIGMIPS = False

    def teardown(self, mode, network):
        # Revert config
        config.__dict__.update(self.default_config)

    def time_main_complex(self, mode, network):
        # Do it!
        compute.main_complex(self.network, self.state)
