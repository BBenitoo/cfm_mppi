import os
import torch

import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cvxpy as cp
import jax
import jax.numpy as jnp

from cfm_mppi.vrc.build_vrc import (
    RobotForceParameters,
    VRCEllipse,
    force_from_vrc_tube,
)

@jax.jit
def barrier(r_ab, v_rel, dt):
    dist = jnp.linalg.norm(r_ab)
    
    r_ab_pred = r_ab - v_rel * dt
    pred_dist = jnp.linalg.norm(r_ab_pred)
    v_rel_dt_sq = jnp.dot(v_rel * dt, v_rel * dt)
    
    expr = (dist + pred_dist)**2 - v_rel_dt_sq
    b = 0.5 * jnp.sqrt(jnp.maximum(expr, 1e-6))
    return b

@jax.jit
def grad_barrier_exp(r_ab, v_rel, dt):
    def rep_potential(barrier_val, A=2.1, B=0.5):
        return A * jnp.exp(-barrier_val / B)

    V = lambda r: rep_potential(barrier(r, v_rel, dt))
    return jax.grad(V)(r_ab)

class HumanAgent:
    def __init__(self, robot_goal, radius=0.5, dt=0.1, random_generator=None):
        if random_generator is None:
            random_generator = np.random.RandomState()
        
        self.rng = random_generator
        
        while True:
            self.start = self.rng.uniform(-2, 8, size=(2,))
            if np.linalg.norm(self.start) >= 1.5:
                break
        while True:
            self.goal = self.rng.uniform(-2, 8, size=(2,))
            if np.linalg.norm(self.goal - robot_goal) >= 2.0:
                break
                
        self.radius = radius
        self.sfm_des_speed = self.rng.uniform(0.8, 1.3)
        self.dt = dt
        
        self.state = self.start.copy()
        self.control = np.zeros(2, dtype=np.float32)

    def social_force_step(
        self,
        others_states,
        others_controls,
        tau=0.5,
        vrc_tube: list[VRCEllipse] | None = None,
        vrc_current_index: int = 0,
        vrc_force_params: RobotForceParameters | None = None,
        vrc_preview_steps: int = 8,
        vrc_discount: float = 0.85,
    ):
        """Advance one SFM step, optionally reacting to a robot VRC tube."""
        distance_to_goal = np.linalg.norm(self.goal - self.state)
        if distance_to_goal < 0.1 and not vrc_tube:
            self.control = np.zeros(2, dtype=np.float32)
            return

        if distance_to_goal < 0.1:
            desired_velocity = np.zeros(2, dtype=np.float32)
        else:
            goal_direction = (self.goal - self.state) / distance_to_goal
            desired_velocity = self.sfm_des_speed * goal_direction
        
        goal_force = (1/tau) * (desired_velocity - self.control)

        repulsive_force = np.zeros(2, dtype=np.float32)
        for o_x, o_u in zip(others_states, others_controls):

            r_ab = self.state - o_x
            v_rel = self.control - o_u
            
            grad = grad_barrier_exp(jnp.array(r_ab), jnp.array(v_rel), self.dt)
            repulsive_force += -1.0 * np.array(grad)

        vrc_force = np.zeros(2, dtype=np.float32)
        if vrc_tube:
            if vrc_force_params is None:
                vrc_force_params = RobotForceParameters()
            vrc_force = force_from_vrc_tube(
                pedestrian_position=self.state,
                vrc_tube=vrc_tube,
                current_index=min(
                    max(vrc_current_index, 0),
                    len(vrc_tube) - 1,
                ),
                force_params=vrc_force_params,
                preview_steps=vrc_preview_steps,
                discount=vrc_discount,
            ).astype(np.float32)

        total_acceleration = goal_force + repulsive_force + vrc_force
        
        self.control += total_acceleration * self.dt
        
        speed = np.linalg.norm(self.control)
        if speed > self.sfm_des_speed * 1.3:
             self.control = (self.control / speed) * (self.sfm_des_speed * 1.3)

        self.state += self.control * self.dt



class AgentHistory:
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.data = None
        
    def update(self, new_data: torch.Tensor):
        new_data = new_data.unsqueeze(-1)  # [..., D] -> [..., D, 1]
        if self.data is None:
            self.data = new_data
        else:
            if len(self) < self.max_length:
                self.data = torch.cat([self.data, new_data], dim=-1)
            else:
                self.data = torch.cat([self.data[..., 1:], new_data], dim=-1)
                
    def get(self):
        return self.data
        
    def __len__(self):
        return self.data.shape[-1] if self.data is not None else 0
