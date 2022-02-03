from nums.core.array.blockarray import BlockArray


# pylint: disable=import-outside-toplevel
def test_modin(nps_app_inst):
    import nums
    import nums.numpy as nps
    import modin.pandas as mpd
    from nums.core import settings
    from nums.core.systems.systems import RaySystem

    if not isinstance(nps_app_inst.cm.system, RaySystem):
        return

    filename = settings.pj(
        settings.project_root, "tests", "core", "storage", "test.csv"
    )
    ba1 = nums.read_csv(filename, has_header=True)
    df = mpd.read_csv(filename)
    ba2: BlockArray = nums.from_modin(df)
    assert nps.allclose(ba1, ba2)
