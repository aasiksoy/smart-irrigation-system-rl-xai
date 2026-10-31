from simulator.soils import make_soils


def test_soils_are_physically_valid():
    soils = make_soils(1000, seed=0)
    for s in soils:
        assert ..., f"invalid soil: {s}"