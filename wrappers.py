import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)


class ExposeTimeLimit(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, act):
        obs, reward, done, info = self.env.step(act)
        return obs, reward, done, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps

    @property
    def was_real_done(self):
        return self.env.was_real_done

    @was_real_done.setter
    def was_real_done(self, was_real_done):
        self.env.was_real_done = was_real_done

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, act):
        return self.env.step(act)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()

        if lives < self.lives and lives > 0:
            done = True

        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class EpisodicLifeEnvPong(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.was_real_done = True

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        if reward == -1:
            done = True
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        return obs


class StickyActions(gym.Wrapper):
    def __init__(self, env, frame_skip):
        gym.Wrapper.__init__(self, env)
        self._skip = frame_skip

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, frame_skip):
        gym.Wrapper.__init__(self, env)
        self.obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = frame_skip

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self.obs_buffer[0] = obs
            if i == self._skip - 1: self.obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self.obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.last_reward = None

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps

    @property
    def was_real_done(self):
        return self.env.was_real_done

    @was_real_done.setter
    def was_real_done(self, was_real_done):
        self.env.was_real_done = was_real_done

    def reward(self, reward):
        self.last_reward = reward
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, frame_size, grayscale=True):
        super().__init__(env)
        width, height = frame_size
        self._width = width
        self._height = height
        self._grayscale = grayscale
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(low=0,
                                   high=255,
                                   shape=(self._height, self._width, num_colors),
                                   dtype=np.uint8)

        original_space = self.observation_space
        self.observation_space = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps

    def observation(self, obs):
        if self._grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, -1)
        obs = cv2.resize(obs, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return obs


class FrameActionStack(gym.Wrapper):
    def __init__(self, env, stack_frames):
        gym.Wrapper.__init__(self, env)
        self.k = 2 * stack_frames
        self.frames = deque([], maxlen=self.k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.k, *shp[:-1]), dtype=env.observation_space.dtype)

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            act_plane = np.full_like(obs, 0)
            act_plane[0, :] = 1
            self.frames.append(act_plane)
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        act_plane = np.full_like(obs, 255*(action/self.env.action_space.n), dtype=np.uint8)
        self.frames.append(act_plane)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class AtariFrameStack(gym.Wrapper):
    def __init__(self, env, stack_frames):
        gym.Wrapper.__init__(self, env)
        self.k = stack_frames
        self.frames = deque([], maxlen=self.k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.k, shp[:-1]), dtype=env.observation_space.dtype)

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class StackFrames(gym.Wrapper):
    def __init__(self, env, stack_frames):
        gym.Wrapper.__init__(self, env)
        self.k = stack_frames
        self.frames = deque([], maxlen=self.k)

        old_shape = env.observation_space.shape
        if len(old_shape) > 1:
            env.observation_space.shape = (self.k, *old_shape[:-1])
        else:
            env.observation_space.shape = (self.k, *old_shape)

    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps

    @_elapsed_steps.setter
    def _elapsed_steps(self, steps):
        self.env._elapsed_steps = steps

    @property
    def was_real_done(self):
        return self.env.was_real_done

    @was_real_done.setter
    def was_real_done(self, was_real_done):
        self.env.was_real_done = was_real_done

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
        

class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.array(self._frames)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


def wrap_atari(env, config):
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, config.noop_max)
    env = MaxAndSkipEnv(env, config.frame_skip)

    if config.episode_life:
        env = EpisodicLifeEnv(env)

    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env, config.frame_size)

    if config.stack_obs:
        if config.stack_actions:
            env = FrameActionStack(env, config.stack_obs)
        else:
            env = AtariFrameStack(env, config.stack_obs)

    if config.clip_rewards:
        env = ClipRewardEnv(env)

    return env

def wrap_game(env, config):
    if config.wrap_atari:
        env = wrap_atari(env, config)
    else:
        if config.noop_reset:
            env = NoopResetEnv(env, config.noop_max)
        if config.sticky_actions > 1:
            env = StickyActions(env, config.sticky_actions)
        if config.episode_life:
            if 'Pong' in config.environment:
                env = EpisodicLifeEnvPong(env)
            else:
                env = EpisodicLifeEnv(env)
        if config.fire_reset:
            env = FireResetEnv(env)
        if config.stack_obs > 1:
            env = StackFrames(env, config.stack_obs)
        if config.clip_rewards:
            env = ClipRewardEnv(env)

    if not hasattr(env, 'legal_actions'):
      legal_actions = range(env.action_space.n)
      env.legal_actions = lambda: legal_actions

    return env



