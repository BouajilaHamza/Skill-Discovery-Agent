"""Register MiniGrid environments with Gymnasium."""
from gymnasium.envs.registration import register

# Register all available MiniGrid environments
register(
    id="MiniGrid-Empty-5x5-v0",
    entry_point="minigrid.envs:EmptyEnv",
    kwargs={"size": 5}
)

register(
    id="MiniGrid-Empty-8x8-v0",
    entry_point="minigrid.envs:EmptyEnv",
    kwargs={"size": 8}
)

register(
    id="MiniGrid-Empty-16x16-v0",
    entry_point="minigrid.envs:EmptyEnv",
    kwargs={"size": 16}
)

# You can add more environments as needed
register(
    id="MiniGrid-DoorKey-5x5-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 5}
)

register(
    id="MiniGrid-DoorKey-8x8-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 8}
)
