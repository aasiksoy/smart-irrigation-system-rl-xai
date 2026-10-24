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
