# %%
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import navix as nx
import navix.environments.hi_core

# Create the environment
env = nx.make('Hi_Core_task_2', observation_fn=nx.observations.rgb)
key = jax.random.PRNGKey(666)
timestep = env.reset(key)

def render(obs, title):
    plt.imshow(obs)
    plt.title(title)
    plt.axis('off')
    plt.show()

print(timestep.observation.shape)
render(timestep.observation, "Initial observation")
# %%
def unroll(key, num_steps=5):
    timestep = env.reset(key)
    actions = jax.random.randint(key, (num_steps,), 0, env.action_space.n)

    steps = [timestep]
    for action in actions:
        timestep = env.step(timestep, action)
        steps.append(timestep)

    return steps

# Unroll and print steps
steps = unroll(key, num_steps=5)
render(steps[-1].observation, "Last observation")

# %%
@jax.jit
def env_step_jit(timestep, action):
    return env.step(timestep, action)

def unroll_jit_step(key, num_steps=10):
    timestep = env.reset(key)
    actions = jax.random.randint(key, (num_steps,), 0, env.action_space.n)

    steps = [timestep]
    for action in actions:
        timestep = env_step_jit(timestep, action)
        steps.append(timestep)

    return steps

# Example usage
steps = unroll_jit_step(key, num_steps=10)
render(steps[-1].observation, "Last observation")
# %%
%timeit -n 1 -r 3 unroll(key, num_steps=10)
# %%
%timeit -n 1 -r 3 lambda: unroll_jit_step(key, num_steps=10)[-1].block_until_ready()
# %%
def unroll_scan(key, num_steps=10):
    timestep = env.reset(key)
    actions = jax.random.randint(key, (num_steps,), 0, env.action_space.n)

    timestep, _ = jax.lax.scan(
        lambda timestep, action: (env.step(timestep, action), ()),
        timestep,
        actions,
        unroll=10,
    )
    return timestep


# Example usage
unroll_jit_loop = jax.jit(unroll_scan, static_argnums=(1,))
timestep = unroll_jit_loop(key, num_steps=10)
render(timestep.observation, "Last observation")
# %%
# Let's compile the function ahead of time
num_envs = 32
keys = jax.random.split(key, num_envs)
unroll_batched = jax.jit(jax.vmap(unroll_scan, in_axes=(0, None)), static_argnums=(1,)).lower(keys, 10).compile()
# %%
# and run it
last_steps = unroll_batched(keys)
render(last_steps.observation[0], "Last observation of env 0")
print("Batch size of the results", last_steps.reward.shape[0])
# %%
for env_id in nx.registry():
    print(env_id)

# %%
