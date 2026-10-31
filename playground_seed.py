import numpy as np

# --- Version 1: seed INSIDE the function (wrong) ---
def roll_wrong():
    rng = np.random.default_rng(seed=0)     # new dice, same seed, every call
    return rng.uniform(0, 1)

print("Wrong: ", [round(roll_wrong(), 2) for _ in range(3)])


# --- Version 2: seed ONCE, generator passed in (right) ---
def roll_right(rng):
    return rng.uniform(0, 1)                # just use the dice it was given

rng = np.random.default_rng(seed=0)         # dice made ONCE
print("Right: ", [round(roll_right(rng), 2) for _ in range(3)])


# --- Run version 2 again with the same seed ---
rng = np.random.default_rng(seed=0)
print("Again: ", [round(roll_right(rng), 2) for _ in range(3)])