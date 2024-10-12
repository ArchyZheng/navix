# %%
from typing import Union
import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from .. import rewards, observations, terminations
from ..components import EMPTY_POCKET_ID
from ..rendering.cache import RenderingCache
from ..rendering.registry import PALETTE
from . import Environment
from ..entities import Player, Key, Door, Goal, Wall
from ..states import State
from . import Timestep
from ..grid import mask_by_coordinates, room, random_positions, random_directions, random_colour
from .registry import register_env


class Hi_Core_task_0(Environment):
    random_start: bool = struct.field(pytree_node=False, default=False)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        # check minimum height and width
        assert (
            self.height > 3
        ), f"Room height must be greater than 3, got {self.height} instead"
        assert (
            self.width > 4
        ), f"Room width must be greater than 5, got {self.width} instead"

        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        grid = room(height=self.height, width=self.width)

        # door positions
        # col can be between 1 and height - 2
        # door_col = jax.random.randint(k4, (), 2, self.width - 2)  # col
        door_col = jnp.asarray(5)
        # row can be between 1 and height - 2
        door_row = jax.random.randint(k3, (), 1, self.height - 1)  # row
        door_pos = jnp.asarray((door_row, door_col))
        doors = Door.create(
            position=door_pos,
            requires=jnp.asarray(3),
            open=jnp.asarray(False),
            colour=PALETTE.YELLOW,
        )

        # wall positions
        wall_rows = jnp.arange(1, self.height - 1)
        wall_cols = jnp.asarray([door_col] * (self.height - 2))
        wall_pos = jnp.stack((wall_rows, wall_cols), axis=1)
        # remove wall where the door is
        wall_pos = jnp.delete(
            wall_pos, door_row - 1, axis=0, assume_unique_indices=True
        )
        walls = Wall.create(position=wall_pos)

        # get rooms
        first_room_mask = mask_by_coordinates(
            grid, (jnp.asarray(self.height), door_col), jnp.less
        )
        first_room = jnp.where(first_room_mask, grid, -1)  # put walls where not mask
        second_room_mask = mask_by_coordinates(
            grid, (jnp.asarray(0), door_col), jnp.greater
        )
        second_room = jnp.where(second_room_mask, grid, -1)  # put walls where not mask

        # set player and goal pos
        if self.random_start:
            player_pos = random_positions(k1, first_room)
            player_dir = random_directions(k2)
            goal_pos = random_positions(k2, second_room)
        else:
            player_pos = jnp.asarray([1, 1])
            player_dir = jnp.asarray(0)
            goal_pos = jnp.asarray([self.height - 2, self.width - 2])

        # spawn goal and player
        player = Player.create(
            position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID
        )
        goals = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        # spawn key
        key_pos = random_positions(k2, first_room, exclude=player_pos)
        keys = Key.create(position=key_pos, id=jnp.asarray(3), colour=PALETTE.YELLOW)

        # remove the wall beneath the door
        grid = grid.at[tuple(door_pos)].set(0)

        entities = {
            "player": player[None],
            "key": keys[None],
            "door": doors[None],
            "goal": goals[None],
            "wall": walls,
        }

        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )

class Hi_Core_task_1(Environment):
    random_start: bool = struct.field(pytree_node=False, default=False)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        # check minimum height and width
        assert (
            self.height > 3
        ), f"Room height must be greater than 3, got {self.height} instead"
        assert (
            self.width > 4
        ), f"Room width must be greater than 5, got {self.width} instead"

        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        grid = room(height=self.height, width=self.width)

        # door positions
        # col can be between 1 and height - 2
        doors = []
        doors_row_list = [3, 6]
        for i in range(2):
            door_row = jnp.asarray(doors_row_list[i])
            # row can be between 1 and height - 2
            door_col = jax.random.randint(k3, (), 1, self.height - 1)  # row
            door_pos = jnp.asarray((door_row, door_col))
            doors.append(Door.create(
                position=door_pos,
                requires=jnp.asarray(3),
                open=jnp.asarray(False),
                colour=random_colour(k1),
            ))

        # get rooms
        first_room_mask = mask_by_coordinates(
            grid, (jnp.asarray(self.height), door_col), jnp.less
        )
        first_room = jnp.where(first_room_mask, grid, -1)  # put walls where not mask
        second_room_mask = mask_by_coordinates(
            grid, (jnp.asarray(0), door_col), jnp.greater
        )
        second_room = jnp.where(second_room_mask, grid, -1)  # put walls where not mask

        # set player and goal pos
        if self.random_start:
            player_pos = random_positions(k1, first_room)
            player_dir = random_directions(k2)
            goal_pos = random_positions(k2, second_room)
        else:
            player_pos = jnp.asarray([1, 1])
            player_dir = jnp.asarray(0)
            goal_pos = jnp.asarray([self.height - 2, self.width - 2])

        # spawn goal and player
        player = Player.create(
            position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID
        )
        goals = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        # spawn key
        key_pos = random_positions(k2, first_room, exclude=player_pos)
        keys = Key.create(position=key_pos, id=jnp.asarray(3), colour=PALETTE.YELLOW)

        # remove the wall beneath the door
        grid = grid.at[tuple(door_pos)].set(0)
        doors = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *doors)

        entities = {
            "player": player[None],
            "key": keys[None],
            "door": doors[None],
            "goal": goals[None],
        }

        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )
 
register_env(
    "Hi_Core_task_0",
    lambda *args, **kwargs: Hi_Core_task_0.create(
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        height=10,
        width=10,
        random_start=True,
        *args,
        **kwargs,
    ),
)

register_env(
    "Hi_Core_task_1",
    lambda *args, **kwargs: Hi_Core_task_1.create(
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        height=10,
        width=10,
        random_start=False,
        *args,
        **kwargs,
    ),
)
# %%
