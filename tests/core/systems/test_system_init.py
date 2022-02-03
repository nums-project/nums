import ray

from nums.core.systems.systems import RaySystem
from nums.core.systems import utils as systems_utils
from nums.core import settings

# pylint: disable=protected-access


def test_head_detection():
    ray.init()

    assert settings.head_ip is None
    sys = RaySystem(use_head=True)
    sys.init()
    assert sys._head_node is not None
    sys.shutdown()

    settings.head_ip = "1.2.3.4"
    sys = RaySystem(use_head=True)
    sys.init()
    assert sys._head_node is None
    sys.shutdown()

    settings.head_ip = systems_utils.get_private_ip()
    sys = RaySystem(use_head=True)
    sys.init()
    assert sys._head_node is not None
    sys.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    test_head_detection()
