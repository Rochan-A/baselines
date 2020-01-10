import numpy as np
import os, copy
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
from .wrappers import TimeLimit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from skimage import segmentation
from skimage.measure import compare_ssim as ssim
import skimage.color as color

NUM_OBJECTS = 3		# MODIFIED FOR EXTERNAL OBJECT SET
NUM_DUP = 1		# MODIFIED FOR EXTERNAL OBJECT SET

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class Usegm(nn.Module):
    def __init__(self, input_dim, nChannel, nConv):
        super(Usegm, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = []
        self.bn2 = []
        self.nConv = nConv
        for i in range(nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

# Initialize default weights to the frame segmentation network
def init_weights(m):
    if type(m) == nn.Conv2d:
        m.weight.data.fill_(torch.tensor(0.5))
        m.bias.data.fill_(torch.tensor(0))

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True,
                    dict_space_key=None, use_rgb=False, use_segm=False,
                    use_unsup_segm=False):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key

        self.use_rgb = use_rgb
        self.use_segm = use_segm
        self.use_unsup_segm = use_unsup_segm

        if use_unsup_segm:
            self.segm_model = Usegm(3, 100, 3)
            self.optimizer_USOD = optim.SGD(self.segm_model.parameters(), lr=0.1, momentum=0.9)

            self.count = 0

        if self.use_rgb or self.use_segm or self.use_unsup_segm:
            num_colors = 3
            self._grayscale = False
        else:
            num_colors = 1

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )

        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        frame_g = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )

        if not self.use_rgb:

            if self.use_segm:
                frame_slic = segmentation.slic(frame_g, n_segments=5)
                frame_g = color.label2rgb(frame_slic, frame_g, kind='avg')

# TODO
#            if self.use_unsup_segm:
#                if self.count == 5:
#                    self.segm_model.apply(init_weights)
#                    optimizer_USOD.zero_grad()
#                    self.count = 0
#                self.count+=1
#
#                # Scale the image x4 times
#                r = cv2.resize(frame_g, \
#                        (frame_g.shape[1]*4, frame_g.shape[0]*4), \
#                        interpolation=cv2.INTER_NEAREST)
#
#                st = im_preprocess(r[:][:][:], 10, 1000)
#
#                # Segment the frame
#                im_target_rgb = train_segmentation(self.segm_model, st, \
#                        loss_fn, optimizer_USOD, \
#                        label_colours, args)

            if self._grayscale:
                frame_g = cv2.cvtColor(frame_g, cv2.COLOR_RGB2GRAY)
                frame_g = np.expand_dims(frame_g, -1)

        if self._key is None:
            obs = frame_g
        else:
            obs = obs.copy()
            obs[self._key] = frame_g
        return obs

class WarpFrame_feov(gym.ObservationWrapper):
    def __init__(self, env, num_objects=None, object_set=None, dict_space_key=None):
        """
        NOTE: Use Feature Extraction for Object Detection to generate vector
              space.

        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._key = dict_space_key
        self._num_objects = num_objects

        #assert object_set != None
        #self.object_set = np.load(object_set, allow_pickle=True)

        new_space = gym.spaces.Box(
            low=0,
            high=180,
            shape=(6, ),
            dtype=np.uint8,
        )

        #self.frame_his = np.zeros((4, self._num_objects*2), dtype=np.uint8)

        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        vector = np.zeros((6), dtype=np.uint8)

        ob = frame[40:190]

        image_ = ob.reshape(-1, ob.shape[-1])
        pix_values, _, _ = np.unique(image_, axis=0, return_index=True, return_counts = True)

        _, counts = np.unique(image_, axis=0, return_counts = True)

        #[[144  72  17]
        # [213 130  74]
        # [236 236 236]]

        unique = np.delete(pix_values, np.where(counts == max(counts)), axis= 0)

        # Extract the location using the above pixel values
        # Array to store cordinates of boxes
        objs = []
        objects = []

        for i, peak in enumerate(unique):
            # First we create a mask selecting all the pixels of this unqiue color
            mask = cv2.inRange(ob, peak, peak)

            # And use it to extract the corresponding part of the original colour image
            blob = cv2.bitwise_and(ob, ob, mask=mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for j, contour in enumerate(contours):
                bbox = cv2.boundingRect(contour)
                # Create a mask for this contour
                contour_mask = np.zeros_like(mask)
                cv2.drawContours(contour_mask, contours, j, 255, -1)

                # Extract the pixels belonging to this contour
                result = cv2.bitwise_and(blob, blob, mask=contour_mask)
                # And draw a bounding box
                top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])

                # Save the box to a list (only if box size is greater than 8x8)
                if bottom_right[1] - top_left[1] > 2 and bottom_right[0] - top_left[0] > 2:
                    objs.append([top_left, bottom_right])

                # Stuff to draw boxes and save the file
                    cv2.rectangle(result, top_left, bottom_right, (0, 0, 255), 1)
                    #file_name_bbox = "blobs-%d-(%03d-%03d-%03d)_%d-bbox.png" % (i, peak[0], peak[1], peak[2], j)
                    #cv2.imwrite(file_name_bbox, result)
                    cv2.imshow("Frame", result)
                    cv2.waitKey(10)

        imc = copy.copy(ob)

        for j, contour in enumerate(objs):
            objects.append(ob[contour[0][1]:contour[1][1], contour[0][0]:contour[1][0]])

        o_objects = len(objects)

        # Check for duplicate objects
        for i, obj in enumerate(objects):
            try:
                # Take care of templates that are the same but of different sizes
                for k in range(i + 1, len(objects)):
                    _, newx = max(enumerate([obj.shape[0], objects[k].shape[0]]))
                    _, newy = max(enumerate([obj.shape[1], objects[k].shape[1]]))

                    obj1 = cv2.resize(objects[k],(int(newx),int(newy)))
                    obj2 = cv2.resize(obj,(int(newx),int(newy)))

                    _, minxy = min(enumerate([obj1.shape[0], obj1.shape[1]]))

                    ratio = int(8/minxy)
                    if ratio > 1:
                        obj2 = cv2.resize(obj2,(obj2.shape[1]*ratio, obj2.shape[0]*ratio))
                        obj1 = cv2.resize(obj1,(obj1.shape[1]*ratio, obj1.shape[0]*ratio))

                    res = ssim(obj1, obj2, multichannel=True)
                    if res >= 0.7:
                        objects.pop(k)
                        objs.pop(k)
            except:
                b = 0

        for j, contour in enumerate(objs):
            vector[j*2] = (contour[0][0] + contour[1][0])/2
            vector[j*2 + 1] = (contour[0][1] + contour[1][1])/2

        if self._key is None:
            obs = vector
        else:
            obs = obs.copy()
            obs[self._key] = vector
        return obs

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
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

def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False,
                            scale=False, use_rgb=False, use_segm=False,
                            use_unsup_segm=False, use_feov=False,
                            num_objects=None, object_set=None):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    if use_feov:
        env = WarpFrame_feov(env, num_objects=num_objects, object_set=object_set)
    elif (use_rgb or use_segm or use_unsup_segm):
        env = WarpFrame(env, use_rgb=use_rgb, use_segm=use_segm,
                use_unsup_segm=use_unsup_segm)
    else:
        env = WarpFrame(env)

    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

