import psutil

from nums.core.compute import numpy_compute
from nums.core.systems import utils as systems_utils


def test_utils():
    r = systems_utils.get_module_functions(systems_utils)
    assert len(r) > 0
    r = systems_utils.get_instance_functions(numpy_compute.ComputeCls())
    assert len(r) > 0


def test_num_cpus():
    all_cores = psutil.cpu_count(logical=False)
    returned_cores = systems_utils.get_num_cores(reserved_for_os=0)
    assert all_cores == returned_cores
    returned_cores = systems_utils.get_num_cores(reserved_for_os=2)
    if all_cores <= returned_cores:
        # CI machines may have few cores.
        assert all_cores == returned_cores
    else:
        assert all_cores - 2 == returned_cores


if __name__ == "__main__":
    # pylint: disable=import-error
    test_utils()
    test_num_cpus()
