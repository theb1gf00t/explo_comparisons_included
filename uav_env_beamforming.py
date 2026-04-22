"""
uav_env_beamforming.py

Adapted environment for the M2LLM-Driven DRL paper comparison baseline.

Changes from uav_env_5.py:
  - Each UAV action is 3-dimensional: [vx, vy, intensity]
    where intensity ∈ [-1, 1] maps to treatment resource allocation [0, 1].
  - Treatment speed scales with intensity:
      effective_days = max(1, round(TREATMENT_DAYS / max(intensity, 0.1)))
    (higher intensity → fewer days → faster cure — analogous to higher beamforming
    power → better channel quality in the original paper).
  - step() accepts either:
      (a) a list of N_UAVS arrays of shape (3,), or
      (b) a single flat array of shape (N_UAVS * ACTION_DIM,)  ← centralized DDPG
  - Treatment quality bonus added to reward:
      TREATMENT_QUALITY_SCALE * intensity_scaled  (when treatment starts)
  - Disease dynamics, reward shaping, observations: identical to uav_env_5.
"""

import math
import numpy as np

# ─── GRID / EPISODE CONSTANTS ─────────────────────────────────────────────────

GRID_ROWS  = 10
GRID_COLS  = 10
N_SECTORS  = GRID_ROWS * GRID_COLS
N_UAVS     = 4
T_MAX      = 72

# ─── UAV PHYSICS ──────────────────────────────────────────────────────────────

E_MAX           = 150.0
E_MOVE          = 1.0
E_HOVER         = 1.21
TAU_DIAG        = 2
DAILY_STEPS_MAX = 80
HOVER_MAG       = 0.2

# ─── CRASH / RETURN CONSTANTS ─────────────────────────────────────────────────

RETURN_BUFFER             = 5
SURVIVAL_RATIO_THRESHOLD  = 1.5

# ─── DISEASE SPREAD ───────────────────────────────────────────────────────────

HEALING_PERIOD  = 28
SPREAD_ALPHA    = 0.015
SPREAD_BETA     = 0.026
RESEED_PROB     = 0.04
RESEED_DAY_MIN  = 10
RESEED_DAY_MAX  = 50
INIT_SEEDS_MIN  = 2
INIT_SEEDS_MAX  = 5

_CORNER_SIDS = {0, GRID_COLS - 1, N_SECTORS - GRID_COLS, N_SECTORS - 1}

# ─── RISK / REWARD PARAMETERS ─────────────────────────────────────────────────

GAMMA      = 0.8
ETA        = 0.03
ALPHA      = 0.4
SIGMA      = 2.0
H_MAX      = 10.0

PSI             = 1.0
LAMBDA_ENG      = 0.1
ZETA            = 1.0
SIGMA_REP       = 2.0
EPSILON         = 1.0
W_UNKNOWN_FLOOR = 0.1

# ─── TREATMENT / IMMUNITY ─────────────────────────────────────────────────────

TREATMENT_DAYS         = 3
IMMUNITY_DAYS          = 5
TREATMENT_QUALITY_SCALE = 5.0   # reward bonus per treatment start, scaled by intensity

# ─── UNIFIED REWARD ECONOMY ───────────────────────────────────────────────────

INFECTED_FOUND_BONUS      = 30.0   # matched to uav_env_5 for fair comparison
NEW_OUTBREAK_MULTIPLIER   = 2.0
HEALTHY_FOUND_BONUS       = 3.0
CRASH_PENALTY             = 20.0
END_OF_DAY_CRASH_PENALTY  = 5.0
SAFE_RETURN_BONUS         = 5.0
OVERHOVER_PENALTY         = 0.3
DIAGNOSED_INFECTED_DECAY  = 0.1
RETURN_POTENTIAL_SCALE    = 6.0
STEP_ALIVE_REWARD         = 0.01

# ─── OBSERVATION / ACTION SIZES ───────────────────────────────────────────────

ENV_OBS_DIM      = 5
OBS_SIZE         = 9 + ENV_OBS_DIM + N_SECTORS + N_SECTORS + (N_UAVS - 1) * 2  # 220
JOINT_SIZE       = N_UAVS * OBS_SIZE                                            # 880
ACTION_DIM       = 3              # [vx, vy, intensity]
JOINT_ACTION_DIM = N_UAVS * ACTION_DIM                                          # 12


class UAVFieldEnvBeamforming:
    """
    10x10 multi-UAV crop disease monitoring.
    Extended with per-UAV treatment intensity (beamforming analogue).
    """

    def __init__(self, seed=None):
        self.T = T_MAX

        self.sector_pos = {
            sid: (sid // GRID_COLS, sid % GRID_COLS)
            for sid in range(N_SECTORS)
        }
        self.pos_to_sid = {v: k for k, v in self.sector_pos.items()}
        self.neighbors  = self._build_neighbors()

        self.sector_rows = np.arange(N_SECTORS, dtype=np.float32) // GRID_COLS
        self.sector_cols = np.arange(N_SECTORS, dtype=np.float32) %  GRID_COLS
        self.sector_rc   = np.stack([self.sector_rows, self.sector_cols], axis=1)

        self._two_sigma2     = 2.0 * SIGMA     ** 2
        self._two_sigma_rep2 = 2.0 * SIGMA_REP ** 2

        self.ap_pos = [
            (0.0, 0.0),
            (0.0, float(GRID_COLS - 1)),
            (float(GRID_ROWS - 1), 0.0),
            (float(GRID_ROWS - 1), float(GRID_COLS - 1)),
        ]

        self._candidate_sids = np.array(
            [sid for sid in range(N_SECTORS) if sid not in _CORNER_SIDS],
            dtype=np.int32
        )

        self.rng = np.random.default_rng(seed)
        self.last_reward_components = [{} for _ in range(N_UAVS)]
        self.reset()

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self):
        self.current_day = 0
        self.daily_step  = 0

        n_seeds = int(self.rng.integers(INIT_SEEDS_MIN, INIT_SEEDS_MAX))
        init_seeds = self.rng.choice(
            self._candidate_sids, size=n_seeds, replace=False
        )

        self.true_status = np.zeros(N_SECTORS, dtype=np.int8)
        self.true_status[init_seeds] = 1

        self.healing_timer = np.zeros(N_SECTORS, dtype=np.int16)
        self.healing_timer[init_seeds] = HEALING_PERIOD

        self.wind_base_dir = float(self.rng.uniform(0, 360))
        self._generate_env_vars(0)

        self.uav_status  = np.full(N_SECTORS, 2, dtype=int)
        self.H           = np.zeros(N_SECTORS, dtype=float)
        self.last_visit  = np.zeros(N_SECTORS, dtype=int)

        self.uav_pos = [
            (0.0, 0.0),
            (0.0, float(GRID_COLS - 1)),
            (float(GRID_ROWS - 1), 0.0),
            (float(GRID_ROWS - 1), float(GRID_COLS - 1)),
        ]
        self.last_v         = [np.zeros(2, dtype=np.float32) for _ in range(N_UAVS)]
        self.last_intensity = np.zeros(N_UAVS, dtype=np.float32)

        self.energy = [float(E_MAX)] * N_UAVS
        self.dwell  = [0] * N_UAVS

        self.crashed             = [False] * N_UAVS
        self.newly_crashed       = [False] * N_UAVS
        self.safely_returned     = [False] * N_UAVS
        self.left_ap_today       = [False] * N_UAVS
        self._end_of_day_crash   = [False] * N_UAVS

        self.treatment_timer    = np.zeros(N_SECTORS, dtype=int)
        self.intervention_timer = np.zeros(N_SECTORS, dtype=int)
        self.intervention_mask  = np.zeros(N_SECTORS, dtype=bool)

        self.ever_diagnosed = np.zeros(N_SECTORS, dtype=bool)
        self.ever_infected  = np.zeros(N_SECTORS, dtype=bool)
        self.ever_infected[init_seeds] = True

        self._newly_diagnosed_by  = {}
        self._treatment_intensity = {}   # sid -> intensity at time of treatment start
        self._ever_infected_before_step  = np.zeros(N_SECTORS, dtype=bool)
        self._ever_diagnosed_before_step = np.zeros(N_SECTORS, dtype=bool)

        self.w = self._compute_risk_weights()
        self.last_phi_explore = [self._phi_explore(u) for u in range(N_UAVS)]
        self.last_phi_return  = [self._phi_return(u)  for u in range(N_UAVS)]

        return self._get_all_obs()

    # ── Step ───────────────────────────────────────────────────────────────────
    def step(self, actions):
        """
        actions: list of N_UAVS arrays of shape (3,) = [vx, vy, intensity],
                 OR flat numpy array of shape (JOINT_ACTION_DIM,).
        """
        if isinstance(actions, np.ndarray) and actions.ndim == 1:
            actions = actions.reshape(N_UAVS, ACTION_DIM)

        assert len(actions) == N_UAVS

        energy_consumed = [0.0] * N_UAVS
        self.newly_crashed       = [False] * N_UAVS
        self.safely_returned     = [False] * N_UAVS
        self._end_of_day_crash   = [False] * N_UAVS
        self._newly_diagnosed_by  = {}
        self._treatment_intensity = {}

        self._ever_infected_before_step  = self.ever_infected.copy()
        self._ever_diagnosed_before_step = self.ever_diagnosed.copy()

        # Decode actions ──────────────────────────────────────────────────────
        for u in range(N_UAVS):
            if self.crashed[u]:
                self.last_intensity[u] = 0.0
                continue

            act = np.array(actions[u], dtype=np.float32)
            v_u = np.clip(act[:2], -1.0, 1.0)
            intensity_raw = float(np.clip(act[2], -1.0, 1.0))
            intensity_scaled = (intensity_raw + 1.0) / 2.0       # → [0, 1]
            intensity_scaled = float(np.clip(intensity_scaled, 0.1, 1.0))
            self.last_intensity[u] = intensity_scaled

            v_mag = float(np.linalg.norm(v_u))
            if v_mag > 1.0:
                v_u   = v_u / v_mag
                v_mag = 1.0

            r, c = self.uav_pos[u]
            nr = np.clip(r + v_u[0], 0, GRID_ROWS - 1)
            nc = np.clip(c + v_u[1], 0, GRID_COLS - 1)
            self.uav_pos[u] = (float(nr), float(nc))
            if not self._at_ap(u):
                self.left_ap_today[u] = True

            if v_mag < HOVER_MAG:
                self.dwell[u]     += 1
                energy_consumed[u] = E_HOVER
            else:
                self.dwell[u]      = 0
                energy_consumed[u] = E_MOVE

            self.energy[u] = max(0.0, self.energy[u] - energy_consumed[u])
            self.last_v[u] = v_u

            if self.energy[u] <= 0 and not self._at_ap(u):
                self.crashed[u]       = True
                self.newly_crashed[u] = True
                self.dwell[u]         = 0
                self.last_v[u]        = np.zeros(2, dtype=np.float32)

        # Diagnosis ───────────────────────────────────────────────────────────
        for u in range(N_UAVS):
            if self.crashed[u]:
                continue

            if self.dwell[u] >= TAU_DIAG:
                r_cont, c_cont = self.uav_pos[u]
                r_int = int(np.clip(round(r_cont), 0, GRID_ROWS - 1))
                c_int = int(np.clip(round(c_cont), 0, GRID_COLS - 1))
                sid = self.pos_to_sid[(r_int, c_int)]

                if self.uav_status[sid] == 2:
                    self.uav_status[sid] = int(self.true_status[sid])
                    self.ever_diagnosed[sid] = True

                    if self.uav_status[sid] == 1:
                        self.ever_infected[sid] = True
                        # Beamforming analogue: higher intensity → fewer treatment days
                        eff_days = max(1, round(TREATMENT_DAYS / self.last_intensity[u]))
                        self.treatment_timer[sid] = eff_days
                        self._treatment_intensity[sid] = self.last_intensity[u]

                    if u not in self._newly_diagnosed_by:
                        self._newly_diagnosed_by[u] = []
                    self._newly_diagnosed_by[u].append(sid)

                b_kt = 1 if (self.true_status[sid] == 1 and
                             self.uav_status[sid] == 1) else 0
                self.H[sid]          = min(H_MAX, GAMMA * self.H[sid] + b_kt)
                self.last_visit[sid] = self.current_day

        self.daily_step += 1

        end_of_day = (self.daily_step >= DAILY_STEPS_MAX)
        if end_of_day:
            self._mark_end_of_day_outcomes()

        self.w = self._compute_risk_weights()
        rewards = [self._compute_reward(u, energy_consumed[u]) for u in range(N_UAVS)]

        if end_of_day:
            self._daily_reset()

        done = (self.current_day >= self.T)
        obs  = self._get_all_obs()

        survival_ratios = [self._survival_ratio(u) for u in range(N_UAVS)]
        info = {
            "t":                self.current_day * DAILY_STEPS_MAX + self.daily_step,
            "current_day":      self.current_day,
            "daily_step":       self.daily_step,
            "uav_pos":          list(self.uav_pos),
            "energy":           list(self.energy),
            "uav_status":       self.uav_status.copy(),
            "true_status":      self.true_status.copy(),
            "risk_weights":     self.w.copy(),
            "dwell":            list(self.dwell),
            "treatment_timer":  self.treatment_timer.copy(),
            "immunity_timer":   self.intervention_timer.copy(),
            "newly_crashed":    list(self.newly_crashed),
            "safely_returned":  list(self.safely_returned),
            "survival_ratio":   survival_ratios,
            "last_intensity":   self.last_intensity.copy(),
            "wind_speed":       self.wind_speed,
            "wind_dir":         self.wind_dir,
            "humidity":         self.humidity,
            "season_mult":      self.season_mult,
            "reward_components": [dict(x) for x in self.last_reward_components],
        }
        return obs, rewards, done, info

    # ── End-of-day ─────────────────────────────────────────────────────────────
    def _mark_end_of_day_outcomes(self):
        for u in range(N_UAVS):
            if not self.crashed[u] and not self._at_ap(u):
                self.crashed[u]           = True
                self.newly_crashed[u]     = True
                self._end_of_day_crash[u] = True

        for u in range(N_UAVS):
            if self._at_ap(u) and not self.crashed[u] and self.left_ap_today[u]:
                self.safely_returned[u] = True

    def _daily_reset(self):
        self.current_day += 1

        if self.current_day <= self.T:
            self._generate_env_vars(self.current_day)
            self._advance_disease()

        active_tx = (self.uav_status == 1) & (self.treatment_timer > 0)
        self.treatment_timer[active_tx] -= 1

        newly_healed = (self.uav_status == 1) & (self.treatment_timer == 0) & active_tx
        if newly_healed.any():
            self.uav_status[newly_healed]         = 0
            self.true_status[newly_healed]        = 0
            self.intervention_timer[newly_healed] = IMMUNITY_DAYS

        imm_active = self.intervention_timer > 0
        self.intervention_timer[imm_active] -= 1
        self.intervention_mask = self.intervention_timer > 0

        re_infected = (self.uav_status == 0) & (self.true_status == 1)
        if re_infected.any():
            self.uav_status[re_infected]         = 2
            self.treatment_timer[re_infected]    = 0
            self.intervention_timer[re_infected] = 0
            self.intervention_mask[re_infected]  = False
            self.healing_timer[re_infected]      = HEALING_PERIOD

        visited_today = (self.last_visit == self.current_day - 1)
        self.H[~visited_today] *= GAMMA

        self.energy              = [float(E_MAX)] * N_UAVS
        for u in range(N_UAVS):
            self.uav_pos[u]      = self.ap_pos[u]
        self.dwell               = [0] * N_UAVS
        self.crashed             = [False] * N_UAVS
        self.last_v              = [np.zeros(2, dtype=np.float32) for _ in range(N_UAVS)]
        self.last_intensity      = np.zeros(N_UAVS, dtype=np.float32)
        self.left_ap_today       = [False] * N_UAVS
        self._end_of_day_crash   = [False] * N_UAVS

        self.w = self._compute_risk_weights()
        self.last_phi_explore = [self._phi_explore(u) for u in range(N_UAVS)]
        self.last_phi_return  = [self._phi_return(u)  for u in range(N_UAVS)]

        self.daily_step = 0

    # ── Environmental variables ─────────────────────────────────────────────────
    def _generate_env_vars(self, day):
        self.wind_speed = float(np.clip(
            5.0 + self.rng.normal(0, 1.5), 1.0, 12.0
        ))
        base_dir = self.wind_base_dir + (90.0 / self.T) * day
        self.wind_dir = float((base_dir + self.rng.normal(0, 10)) % 360)
        self.humidity = float(np.clip(
            60 + 25 * math.sin(math.pi * day / self.T)
            + self.rng.normal(0, 3),
            40, 100
        ))
        self.season_mult = float(np.clip(
            0.8 + 0.5 * math.sin(math.pi * day / self.T),
            0.5, 1.5
        ))

    # ── Disease spread ──────────────────────────────────────────────────────────
    @staticmethod
    def _contact_weight(row_j, col_j, row_k, col_k):
        dr = abs(row_k - row_j); dc = abs(col_k - col_j)
        return 1.0 if (dr + dc == 1) else 0.5

    @staticmethod
    def _wind_alignment(row_j, col_j, row_k, col_k, wind_dir_deg):
        dx = col_k - col_j
        dy = row_j - row_k
        theta_jk = math.degrees(math.atan2(dy, dx))
        diff     = math.radians(wind_dir_deg - theta_jk)
        return max(0.0, math.cos(diff))

    def _compute_spread_prob(self, sid_k, infected_neighbors):
        row_k, col_k = self.sector_pos[sid_k]
        survival = 1.0
        for j in infected_neighbors:
            row_j, col_j = self.sector_pos[j]
            contact_w = self._contact_weight(row_j, col_j, row_k, col_k)
            wind_w    = self._wind_alignment(row_j, col_j, row_k, col_k, self.wind_dir)
            p_j = min(
                (SPREAD_ALPHA * contact_w + SPREAD_BETA * wind_w)
                * (self.humidity / 100.0) * self.season_mult,
                0.95
            )
            survival *= (1.0 - p_j)
        return 1.0 - survival

    def _advance_disease(self):
        new_status = self.true_status.copy()
        new_timer  = self.healing_timer.copy()

        for sid in range(N_SECTORS):
            if self.true_status[sid] == 0:
                if self.intervention_mask[sid]:
                    continue
                inf_nbrs = [n for n in self.neighbors[sid]
                            if self.true_status[n] == 1]
                if inf_nbrs:
                    p = self._compute_spread_prob(sid, inf_nbrs)
                    if self.rng.random() < p:
                        new_status[sid] = 1
                        new_timer[sid]  = HEALING_PERIOD
            elif self.true_status[sid] == 1:
                if self.treatment_timer[sid] > 0:
                    continue
                new_timer[sid] -= 1
                if new_timer[sid] <= 0:
                    new_status[sid] = 0
                    new_timer[sid]  = 0

        if RESEED_DAY_MIN <= self.current_day <= RESEED_DAY_MAX:
            if self.rng.random() < RESEED_PROB:
                pool = np.where(
                    (new_status == 0) & ~self.intervention_mask
                )[0]
                pool = np.array([s for s in pool if s not in _CORNER_SIDS],
                                dtype=np.int32)
                if pool.size > 0:
                    new_sid = int(self.rng.choice(pool))
                    new_status[new_sid] = 1
                    new_timer[new_sid]  = HEALING_PERIOD

        newly_infected = (new_status == 1) & (self.true_status == 0)
        self.ever_infected[newly_infected] = True
        self.true_status   = new_status
        self.healing_timer = new_timer

    # ── Reward ─────────────────────────────────────────────────────────────────
    def _compute_reward(self, u, energy_consumed):
        if self.crashed[u] and not self.newly_crashed[u]:
            return 0.0

        survival_ratio  = self._survival_ratio(u)
        phi_explore_now = self._phi_explore(u)
        phi_return_now  = self._phi_return(u)

        if survival_ratio > SURVIVAL_RATIO_THRESHOLD:
            r_nav    = phi_explore_now - self.last_phi_explore[u]
            nav_mode = "explore"
        else:
            r_nav    = phi_return_now - self.last_phi_return[u]
            nav_mode = "return"

        energy_penalty = LAMBDA_ENG * energy_consumed
        repulsion      = ZETA * self._compute_repulsion(u)
        reward         = r_nav - energy_penalty - repulsion

        discovery_bonus  = self._discovery_bonus(u)
        overhover_pen    = self._overhover_penalty(u)
        reward += discovery_bonus + overhover_pen

        alive_bonus = STEP_ALIVE_REWARD if not self.newly_crashed[u] else 0.0
        reward += alive_bonus

        # Beamforming analogue: bonus for high-intensity treatment application
        tx_quality_bonus = 0.0
        for sid in self._newly_diagnosed_by.get(u, []):
            if self.uav_status[sid] == 1 and sid in self._treatment_intensity:
                tx_quality_bonus += TREATMENT_QUALITY_SCALE * self._treatment_intensity[sid]
        reward += tx_quality_bonus

        crash_penalty = 0.0
        if self.newly_crashed[u]:
            crash_penalty = (END_OF_DAY_CRASH_PENALTY if self._end_of_day_crash[u]
                             else CRASH_PENALTY)
            reward -= crash_penalty

        safe_return_bonus = 0.0
        if self.safely_returned[u]:
            safe_return_bonus = SAFE_RETURN_BONUS
            reward += safe_return_bonus

        self.last_reward_components[u] = {
            "nav_mode":             nav_mode,
            "nav_reward":           float(r_nav),
            "phi_explore":          float(phi_explore_now),
            "phi_return":           float(phi_return_now),
            "survival_ratio":       float(survival_ratio),
            "energy_penalty":       float(energy_penalty),
            "repulsion":            float(repulsion),
            "discovery_bonus":      float(discovery_bonus),
            "tx_quality_bonus":     float(tx_quality_bonus),
            "overhover_penalty":    float(overhover_pen),
            "alive_bonus":          float(alive_bonus),
            "crash_penalty":        float(crash_penalty),
            "end_of_day_crash":     bool(self._end_of_day_crash[u]),
            "safe_return_bonus":    float(safe_return_bonus),
            "total_reward":         float(reward),
        }
        self.last_phi_explore[u] = phi_explore_now
        self.last_phi_return[u]  = phi_return_now
        return reward

    def _phi_explore(self, u):
        r_u, c_u = self.uav_pos[u]
        dists = np.sqrt((self.sector_rows - r_u) ** 2 + (self.sector_cols - c_u) ** 2)
        effective_w = self.w.astype(np.float32).copy()
        diagnosed_infected = (self.uav_status == 1) & self.ever_diagnosed
        effective_w[diagnosed_infected] *= DIAGNOSED_INFECTED_DECAY
        return float(PSI * np.sum(effective_w / (dists + EPSILON)))

    def _phi_return(self, u):
        return float(-RETURN_POTENTIAL_SCALE * self._dist_to_ap(u))

    def _discovery_bonus(self, u):
        bonus = 0.0
        for sid in self._newly_diagnosed_by.get(u, []):
            if self.uav_status[sid] == 1:
                if not self._ever_infected_before_step[sid]:
                    bonus += INFECTED_FOUND_BONUS * NEW_OUTBREAK_MULTIPLIER
                else:
                    bonus += INFECTED_FOUND_BONUS
            elif self.uav_status[sid] == 0:
                if not self._ever_diagnosed_before_step[sid]:
                    bonus += HEALTHY_FOUND_BONUS
        return bonus

    def _overhover_penalty(self, u):
        if self.crashed[u]:
            return 0.0
        r_cont, c_cont = self.uav_pos[u]
        r_int = int(np.clip(round(r_cont), 0, GRID_ROWS - 1))
        c_int = int(np.clip(round(c_cont), 0, GRID_COLS - 1))
        sid = self.pos_to_sid[(r_int, c_int)]

        newly = self._newly_diagnosed_by.get(u, [])
        if sid in newly:
            return 0.0
        if self.dwell[u] > (TAU_DIAG + 1) and self.uav_status[sid] != 2:
            return -OVERHOVER_PENALTY
        return 0.0

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def _survival_ratio(self, u):
        dist         = self._dist_to_ap(u)
        steps_needed = dist + RETURN_BUFFER
        steps_left   = DAILY_STEPS_MAX - self.daily_step
        return steps_left / max(steps_needed, 1e-6)

    def _dist_to_ap(self, u):
        r_u, c_u = self.uav_pos[u]
        r_a, c_a = self.ap_pos[u]
        return np.sqrt((r_u - r_a) ** 2 + (c_u - c_a) ** 2)

    def _at_ap(self, u):
        return self._dist_to_ap(u) < 0.5

    def _compute_risk_weights(self):
        w          = np.zeros(N_SECTORS, dtype=np.float32)
        infected_m = (self.uav_status == 1)
        healthy_m  = (self.uav_status == 0)
        unknown_m  = (self.uav_status == 2)

        w[infected_m] = 1.0

        if healthy_m.any():
            delta_t      = self.current_day - self.last_visit[healthy_m]
            w[healthy_m] = np.minimum(1.0, ETA * delta_t)

        if unknown_m.any():
            omega        = self._compute_omega_batch(unknown_m, infected_m)
            w[unknown_m] = np.maximum(W_UNKNOWN_FLOOR, np.minimum(1.0, omega))
        return w

    def _compute_omega_batch(self, unknown_mask, infected_mask):
        unk_idx = np.where(unknown_mask)[0]
        history = ALPHA * (self.H[unk_idx] / H_MAX)

        inf_idx = np.where(infected_mask)[0]
        if inf_idx.size == 0:
            return history

        diff    = (self.sector_rc[unk_idx, np.newaxis, :]
                   - self.sector_rc[np.newaxis, inf_idx, :])
        dist_sq = (diff ** 2).sum(axis=2)
        spatial = np.exp(-dist_sq / self._two_sigma2).sum(axis=1)
        return history + (1 - ALPHA) * spatial

    def _compute_repulsion(self, u):
        r_u, c_u  = self.uav_pos[u]
        others    = [j for j in range(N_UAVS) if j != u]
        r_others  = np.array([self.uav_pos[j][0] for j in others], dtype=np.float32)
        c_others  = np.array([self.uav_pos[j][1] for j in others], dtype=np.float32)
        dist_sq   = (r_others - r_u) ** 2 + (c_others - c_u) ** 2
        return float(np.sum(np.exp(-dist_sq / self._two_sigma_rep2)))

    # ── Observations ────────────────────────────────────────────────────────────
    def _get_obs(self, u):
        r_u, c_u = self.uav_pos[u]
        r_a, c_a = self.ap_pos[u]
        sr = self._survival_ratio(u)

        own = np.array([
            r_u / (GRID_ROWS - 1),
            c_u / (GRID_COLS - 1),
            self.energy[u] / E_MAX,
            self.last_v[u][0],
            self.last_v[u][1],
            (r_u - r_a) / (GRID_ROWS - 1),
            (c_u - c_a) / (GRID_COLS - 1),
            np.clip(sr, 0, 5) / 5.0,
            self.daily_step / DAILY_STEPS_MAX,
        ], dtype=np.float32)

        env_vars = np.array([
            self.wind_speed / 12.0,
            np.sin(np.radians(self.wind_dir)),
            np.cos(np.radians(self.wind_dir)),
            self.humidity / 100.0,
            (self.season_mult - 0.5) / 1.0,
        ], dtype=np.float32)

        risk        = self.w.astype(np.float32)
        status_norm = (self.uav_status / 2.0).astype(np.float32)

        other_pos = []
        for j in range(N_UAVS):
            if j == u:
                continue
            r_j, c_j = self.uav_pos[j]
            other_pos.append((r_j - r_u) / (GRID_ROWS - 1))
            other_pos.append((c_j - c_u) / (GRID_COLS - 1))
        other_pos = np.array(other_pos, dtype=np.float32)

        return np.concatenate([own, env_vars, risk, status_norm, other_pos])

    def _get_all_obs(self):
        return [self._get_obs(u) for u in range(N_UAVS)]

    def _build_neighbors(self):
        neighbors = {}
        for sid in range(N_SECTORS):
            r, c = self.sector_pos[sid]
            nbrs = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                        nbrs.append(self.pos_to_sid[(nr, nc)])
            neighbors[sid] = nbrs
        return neighbors

    @property
    def total_steps(self):
        return self.T * DAILY_STEPS_MAX
