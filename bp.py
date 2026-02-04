from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import copy
import random
import math


# ----------------------------
# 1) Preference features (Table A.1)
# ----------------------------

BP_FEATURES: Tuple[str, ...] = (
    "plating_ingredients",
    "washing_plates",
    "delivering_dishes",
    "chopping_ingredients",
    "potting_rice",
    "grilling_meat_mushroom",
    "taking_mushroom_from_dispenser",
    "taking_rice_from_dispenser",
    "taking_meat_from_dispenser",
)


@dataclass(frozen=True)
class BPRewardShape:
    """
    Event-based shaping magnitudes.
    The paper’s Table A.1 PDF snippet shows values like (-30, 20) per feature.
    Interpreted as: discouraged = -30, encouraged = +20. Adjust if needed.
    """
    discouraged: float = -30.0
    encouraged: float = 20.0


@dataclass
class BPPreferences:
    """
    preferences[feature] in [-1, +1] (or any real number):
      > 0 => encouraged (adds +encouraged * count)
      < 0 => discouraged (adds discouraged * count)
      = 0 => indifferent
    """
    preferences: Dict[str, float]
    shape: Dict[str, BPRewardShape]

    @staticmethod
    def from_vector(
        vec: List[float],
        features: Tuple[str, ...] = BP_FEATURES,
        default_shape: BPRewardShape = BPRewardShape(),
    ) -> "BPPreferences":
        if len(vec) != len(features):
            raise ValueError(f"Expected vector length {len(features)} but got {len(vec)}")
        prefs = {f: float(v) for f, v in zip(features, vec)}
        shapes = {f: default_shape for f in features}
        return BPPreferences(preferences=prefs, shape=shapes)

    def shaped_reward(self, event_counts: Dict[str, int]) -> float:
        r = 0.0
        for feat, w in self.preferences.items():
            c = int(event_counts.get(feat, 0))
            if c == 0 or w == 0.0:
                continue
            s = self.shape[feat]
            r += (s.encouraged * c) if (w > 0.0) else (s.discouraged * c)
        return r


# ----------------------------
# 2) Event extraction from env "info"
# ----------------------------

class BPEventExtractor:
    """
    Tries to turn env info into {feature_name: count} for one agent index.

    You WILL likely need to adapt EVENT_KEY_MAP to match your environment’s event names.
    """

    # Map our Table A.1 feature names -> a list of event keys to count from env info.
    # If your env uses the same names, keep as-is.
    EVENT_KEY_MAP: Dict[str, List[str]] = {
        "plating_ingredients": ["plating_ingredients"],
        "washing_plates": ["washing_plates"],
        "delivering_dishes": ["delivering_dishes"],
        "chopping_ingredients": ["chopping_ingredients"],
        "potting_rice": ["potting_rice"],
        "grilling_meat_mushroom": ["grilling_meat_mushroom"],
        "taking_mushroom_from_dispenser": ["taking_mushroom_from_dispenser"],
        "taking_rice_from_dispenser": ["taking_rice_from_dispenser"],
        "taking_meat_from_dispenser": ["taking_meat_from_dispenser"],
    }

    @classmethod
    def extract(cls, info: Dict[str, Any], agent_idx: int) -> Dict[str, int]:
        """
        Supported formats (best effort):
          A) info["event_infos"] (Overcooked-AI style):
             - event_infos is a dict: key -> [count_for_agent0, count_for_agent1, ...]
          B) info["events"] is a dict:
             - events is a dict: event_key -> count (already per-agent), OR
             - events is a list/dict per-agent.

        Returns counts for our BP_FEATURES names.
        """
        # --- A) Overcooked-AI style: info["event_infos"]
        if "event_infos" in info and isinstance(info["event_infos"], dict):
            ei = info["event_infos"]  # event_key -> list-of-counts per agent
            return cls._from_event_infos(ei, agent_idx)

        # --- B) info["events"] generic
        if "events" in info:
            ev = info["events"]
            return cls._from_events(ev, agent_idx)

        # --- fallback: no events available
        return {f: 0 for f in BP_FEATURES}

    @classmethod
    def _from_event_infos(cls, event_infos: Dict[str, Any], agent_idx: int) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for feat, keys in cls.EVENT_KEY_MAP.items():
            total = 0
            for k in keys:
                if k not in event_infos:
                    continue
                v = event_infos[k]
                # Typically v is a list like [count_agent0, count_agent1]
                if isinstance(v, (list, tuple)) and agent_idx < len(v):
                    total += int(v[agent_idx])
                elif isinstance(v, int):
                    # If already per-agent count
                    total += int(v)
            out[feat] = total
        return out

    @classmethod
    def _from_events(cls, events_obj: Any, agent_idx: int) -> Dict[str, int]:
        # Case 1: events_obj is dict of counts for this step (assumed already per-agent)
        if isinstance(events_obj, dict):
            return cls._count_keys(events_obj)

        # Case 2: events_obj is list/tuple per agent
        if isinstance(events_obj, (list, tuple)) and agent_idx < len(events_obj):
            per = events_obj[agent_idx]
            if isinstance(per, dict):
                return cls._count_keys(per)

        return {f: 0 for f in BP_FEATURES}

    @classmethod
    def _count_keys(cls, event_dict: Dict[str, Any]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for feat, keys in cls.EVENT_KEY_MAP.items():
            total = 0
            for k in keys:
                if k in event_dict:
                    total += int(event_dict[k])
            out[feat] = total
        return out


# ----------------------------
# 3) A simple preference-driven agent (greedy 1-step lookahead)
# ----------------------------

class PreferenceGreedyAgent:
    """
    Chooses the action that maximizes: base_reward + shaped_reward (from BP events)
    under a 1-step rollout with an assumed partner action.

    Works with multi-agent envs where step() takes a joint action tuple/list.
    """

    def __init__(
        self,
        agent_idx: int,
        prefs: BPPreferences,
        partner_policy: Optional["PreferenceGreedyAgent"] = None,
        explore_eps: float = 0.05,
        use_action_mask: bool = True,
    ):
        self.agent_idx = agent_idx
        self.prefs = prefs
        self.partner_policy = partner_policy
        self.explore_eps = explore_eps
        self.use_action_mask = use_action_mask

    def act(self, env, obs: Any, info: Optional[Dict[str, Any]] = None) -> int:
        # Optional epsilon exploration
        legal_actions = self._legal_actions(env, info, self.agent_idx)
        if not legal_actions:
            return self._random_action(env)

        if random.random() < self.explore_eps:
            return random.choice(legal_actions)

        # If partner policy exists, get its action guess (from current env/obs)
        partner_action = None
        if self.partner_policy is not None:
            partner_obs, partner_info = self._get_partner_view(env)
            partner_action = self.partner_policy._greedy_action(env, partner_obs, partner_info)
        else:
            partner_action = self._random_action(env)  # fallback assumption

        # Evaluate our candidate actions
        best_a = legal_actions[0]
        best_score = -float("inf")
        for a in legal_actions:
            score = self._one_step_score(env, my_action=a, partner_action=partner_action)
            if score > best_score:
                best_score = score
                best_a = a

        return best_a

    def _greedy_action(self, env, obs: Any, info: Optional[Dict[str, Any]] = None) -> int:
        legal_actions = self._legal_actions(env, info, self.agent_idx)
        if not legal_actions:
            return self._random_action(env)
        best_a = legal_actions[0]
        best_score = -float("inf")
        # Partner action is unknown here; assume random
        partner_action = self._random_action(env)
        for a in legal_actions:
            score = self._one_step_score(env, my_action=a, partner_action=partner_action)
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _one_step_score(self, env, my_action: int, partner_action: int) -> float:
        # Build joint action in correct order
        joint = self._make_joint_action(my_action, partner_action)

        # Try to clone env for lookahead
        env2 = self._clone_env(env)
        obs2, base_r, done, trunc, info2 = self._step_env(env2, joint)

        # Convert info -> event counts for this agent
        event_counts = BPEventExtractor.extract(info2 or {}, self.agent_idx)
        shaped_r = self.prefs.shaped_reward(event_counts)

        # You can add small value for "not done" etc. if desired.
        return float(base_r) + shaped_r

    def _make_joint_action(self, my_action: int, partner_action: int):
        # Common convention: joint action as (a0, a1)
        if self.agent_idx == 0:
            return (my_action, partner_action)
        else:
            return (partner_action, my_action)

    @staticmethod
    def _clone_env(env):
        # Prefer env.clone() if available; else deepcopy
        if hasattr(env, "clone") and callable(getattr(env, "clone")):
            return env.clone()
        return copy.deepcopy(env)

    @staticmethod
    def _step_env(env, joint_action):
        # Gymnasium: obs, reward, terminated, truncated, info
        out = env.step(joint_action)
        if len(out) == 5:
            return out
        # Older gym: obs, reward, done, info
        obs, r, done, info = out
        return obs, r, done, False, info

    def _get_partner_view(self, env) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        Best-effort helper. Many multi-agent envs give obs as tuple/list of per-agent obs.
        """
        # If env stores last observation, you can customize this.
        # Here we just return None placeholders.
        return None, None

    def _legal_actions(self, env, info: Optional[Dict[str, Any]], agent_idx: int) -> List[int]:
        """
        If the env provides action masks, use them. Otherwise assume all discrete actions are legal.
        """
        # Try info-based masks first (common pattern)
        if self.use_action_mask and info:
            # A few possible conventions:
            # - info["action_mask"] is [mask_agent0, mask_agent1]
            # - info["action_masks"] similar
            for key in ("action_mask", "action_masks"):
                if key in info:
                    masks = info[key]
                    if isinstance(masks, (list, tuple)) and agent_idx < len(masks):
                        mask = masks[agent_idx]
                        return [i for i, ok in enumerate(mask) if ok]
                    if isinstance(masks, dict) and agent_idx in masks:
                        mask = masks[agent_idx]
                        return [i for i, ok in enumerate(mask) if ok]

        # Fallback: use action space if discrete
        if hasattr(env, "action_space"):
            # Multi-agent envs sometimes use Tuple spaces; keep it simple:
            space = env.action_space
            if hasattr(space, "n"):  # Discrete
                return list(range(space.n))

            # Tuple(Discrete, Discrete)
            if hasattr(space, "spaces") and isinstance(space.spaces, (list, tuple)):
                if agent_idx < len(space.spaces) and hasattr(space.spaces[agent_idx], "n"):
                    return list(range(space.spaces[agent_idx].n))

        return []

    def _random_action(self, env) -> int:
        # Best effort random action
        if hasattr(env, "action_space"):
            space = env.action_space
            if hasattr(space, "sample"):
                try:
                    a = space.sample()
                    if isinstance(a, (list, tuple)):
                        return int(a[self.agent_idx])
                    return int(a)
                except Exception:
                    pass
        return 0


# ----------------------------
# 4) Runner: play an episode with two agents
# ----------------------------

def play_episode(env, agent0: PreferenceGreedyAgent, agent1: PreferenceGreedyAgent, max_steps: int = 10_000):
    obs = env.reset()
    # Gymnasium reset() can return (obs, info)
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, info = obs
    else:
        info = {}

    total_reward = 0.0
    for t in range(max_steps):
        # If obs is per-agent tuple/list, split it; otherwise pass whole obs
        if isinstance(obs, (list, tuple)) and len(obs) >= 2:
            o0, o1 = obs[0], obs[1]
        else:
            o0 = o1 = obs

        a0 = agent0.act(env, o0, info=info)
        a1 = agent1.act(env, o1, info=info)

        step_out = env.step((a0, a1))
        if len(step_out) == 5:
            obs, r, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, r, done, info = step_out
        total_reward += float(r)

        if done:
            break

    return total_reward


# ----------------------------
# 5) Example usage (you must plug in your Overcooked env)
# ----------------------------

if __name__ == "__main__":
    # TODO: create your env here.
    # For example, if you have a gym-registered env:
    #   import gymnasium as gym
    #   env = gym.make("YourOvercookedEnv-v0", layout="open")
    env = None
    raise SystemExit("Please create your environment 'env' before running this script.")

    # Example preference vector:
    # Encourage deliveries + plating, discourage washing; neutral elsewhere.
    vec = [
        +1,  # plating_ingredients
        -1,  # washing_plates
        +1,  # delivering_dishes
        0,   # chopping_ingredients
        0,   # potting_rice
        0,   # grilling_meat_mushroom
        0,   # taking_mushroom_from_dispenser
        0,   # taking_rice_from_dispenser
        0,   # taking_meat_from_dispenser
    ]
    prefs0 = BPPreferences.from_vector(vec)
    prefs1 = BPPreferences.from_vector([0.0] * len(BP_FEATURES))  # partner neutral/randomish

    agent0 = PreferenceGreedyAgent(agent_idx=0, prefs=prefs0, explore_eps=0.02)
    agent1 = PreferenceGreedyAgent(agent_idx=1, prefs=prefs1, explore_eps=0.20)

    # If you want each to assume the other’s greedy action:
    agent0.partner_policy = agent1
    agent1.partner_policy = agent0

    score = play_episode(env, agent0, agent1, max_steps=2400)
    print("Episode score:", score)
