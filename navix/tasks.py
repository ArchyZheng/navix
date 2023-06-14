# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from __future__ import annotations
from typing import Callable


import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from chex import Array

from .components import State
from .grid import mask_entity


def compose(*fns: Callable[[State, Array, State], Array]):
    def composed(prev_state: State, action: Array, state: State) -> Array:
        reward = jnp.asarray(0.0)
        for fn in fns:
            reward += fn(prev_state, action, state)
        return reward

    return composed


def free(state: State) -> Array:
    return jnp.asarray(0.0)


def navigation(
    prev_state: State, action: Array, state: State, prob: ArrayLike = 1.0
) -> Array:
    player_mask = mask_entity(state.grid, state.entities["player/0"].id)
    goal_mask = mask_entity(state.grid, state.entities["goal/0"].id)
    condition = jax.random.uniform(state.key, ()) >= prob
    return jax.lax.cond(
        condition,
        lambda _: jnp.sum(player_mask * goal_mask),
        lambda _: jnp.asarray(0.0),
        (),
    )


def action_cost(
    prev_state: State, action: Array, new_state: State, cost: float = 0.01
) -> Array:
    # noops are free
    return -jnp.asarray(action > 0, dtype=jnp.float32) * cost


def time_cost(
    prev_state: State, action: Array, new_state: State, cost: float = 0.01
) -> Array:
    # time always has a cost
    return -jnp.asarray(cost)


def wall_hit_cost(
    prev_state: State, action: Array, state: State, cost: float = 0.01
) -> Array:
    # if state is unchanged, maybe the wall was hit
    player_before = mask_entity(prev_state.grid, prev_state.entities["player/0"].id)
    player_now = mask_entity(state.grid, state.entities["player/0"].id)
    hit = jnp.array_equal(player_before, player_now)

    # if action is a move action, the wall was hit
    hit *= jnp.less_equal(3, action)
    hit *= jnp.less_equal(action, 6)

    return -jnp.asarray(cost) * hit
