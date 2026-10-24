from dataclasses import dataclass

@dataclass
class SoilParams:
    theta_wp: float 
    theta_fc: float
    theta_sat: float
    drain_rate: float
    root_depth_m: float = 0.6
    p: float = 0.5

    @property
    def w_wp(self) -> float:
        return 1000*self.theta_wp*self.root_depth_m

    @property
    def w_fc(self) -> float:
        return 1000 * self.theta_fc * self.root_depth_m

    @property
    def w_sat(self) -> float:
        return 1000 * self.theta_sat * self.root_depth_m

    #total available water
    @property
    def taw(self) -> float:
        return self.w_fc - self.w_wp

    #readily avaible water
    @property
    def raw(self) -> float:
        return self.p * self.taw

    #below this water level, the crop is stressed
    @property
    def w_stress(self) -> float:
        return self.w_fc - self.raw


#example graph for how kc changes over the seasons:
# Kc
# 1.2 |                 ___________
#     |                /           \
#     |               /             \
# 0.6 |              /               \____
# 0.3 |_____________/
#     +-----|----------|----------|--------|---> day
#     0    20         55         95       125
#      baby  growing    fully grown  ripening
#      plant           (thirstiest)

@dataclass
class CropParams:
    kc_ini: float = 0.3
    kc_mid: float = 1.2 
    kc_end: float = 0.6
    l_ini: int = 20
    l_dev: int = 35
    l_mid: int = 40
    l_late: int = 30

    @property
    def season_length(self)->int:
        return self.l_ini + self.l_dev + self.l_mid + self.l_late

    def kc(self, day: int)-> float:
        if day<self.l_ini:
            return self.kc_ini
        day -= self.l_ini

        if day<self.l_dev:
            return self.kc_ini + (self.kc_mid - self.kc_ini) * day/self.l_dev
        day-= self.l_dev

        if day< self.l_mid:
            return self.kc_mid
        day -= self.l_mid

        if day< self.l_late:
            return self.kc_mid + (self.kc_end - self.kc_mid) * day / self.l_late

        return self.kc_end

def stress_coefficient(w: float, soil: SoilParams) -> float:
    """
    Ks between 0 and 1.
    1.0 = no stress, 0.0 = wilting point.
    """
    if w >= soil.w_stress:
        return 1.0
    ks = (w - soil.w_wp) / (soil.w_stress - soil.w_wp)
    return max(0.0, min(1.0, ks))
#example graph of ks and water usage
# Ks
# 1.0 |              ___________________
#     |             /
# 0.5 |           /
#     |         /
# 0.0 |_______/
#     +-------|-----------|-------------|--> water w (mm)
#            36          66            96
#       wilting point   stress      field capacity
#                      threshold


def step_day(w: float, rain: float, irrigation: float, et0: float,
             day: int, soil: SoilParams, crop: CropParams):
    """
    Advance root-zone water (mm) by one day.
    Returns the new water level and a dict of all flows.
    """

    #1) water coming in
    w = w + rain + irrigation

    #2) our runoff= water above saturation cannot enter the soil
    runoff = max(0.0, w-soil.w_sat)
    w-= runoff

    #3)water above field capacity drains away
    drainage = soil.drain_rate * max(0.0, w-soil.w_fc)
    w-= drainage

    #4) what the crop drinks
    kc = crop.kc(day)
    ks = stress_coefficient(w,soil)
    et_demand = ks*kc*et0
    et_actual = min(et_demand, max(0.0, w - soil.w_wp))
    w -= et_actual

    day_log = {
        "rain": rain,
        "irrigation": irrigation,
        "runoff": runoff,
        "drainage": drainage,
        "et_actual": et_actual,
        "et_potential": kc * et0,
        "kc": kc,
        "ks": ks,
    }
    return w, day_log


def theta_from_w(w: float, soil: SoilParams) -> float:
    """
    Water in mm to volumetric water content (m3/m3)
    """
    return w / (1000 * soil.root_depth_m)