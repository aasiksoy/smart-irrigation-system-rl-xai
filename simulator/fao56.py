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


@dataclass
class CropParams