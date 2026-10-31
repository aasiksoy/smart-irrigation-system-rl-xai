from simulator.soils import make_soils


def test_soils_are_physically_valid():
    #act
    soils = make_soils(1000, seed=0)
    for s in soils:
        #arrange
        wilting_point = s.theta_wp
        field_capacity = s.theta_fc
        saturation = s.theta_sat

        wp_answer =  wilting_point < field_capacity
        fc_answer = field_capacity < saturation
        sat_answer = saturation <= 0.5
        
        #assert
        assert wp_answer, f"wilting point not below field capacity: {s}"
        assert fc_answer, f"field capacity not below saturation: {s}"
        assert sat_answer, f"saturation above 0.5: {s}"


def test_same_seed_gives_same_soils():
    #arrange and act
    soils_a = make_soils(3, seed=2)
    soils_b = make_soils(3, seed=2)

    #assert
    assert soils_a == soils_b, "same seeds but different soils, should have been the opposite"



def test_different_seeds_give_different_soils():
    #arrange and act
    soils_a = make_soils(3, seed=0)
    soils_b = make_soils(3, seed=1)
    
    #assert
    assert soils_a != soils_b, "different seeds but same soils, should have been the opposite"