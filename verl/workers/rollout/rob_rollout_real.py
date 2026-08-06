import os
import time
import torch
import numpy as np
from typing import List, Dict, Any
from verl import DataProto
from tensordict import TensorDict
from .rob_rollout_wm_pro import RobWMHFRolloutPro  # 继承原有逻辑
from verl.utils.libero_utils import resize_image

# 引入新硬件接口
from verl.utils.real_robot.franka_interface import FrankaInterface


class RobWMHFRolloutReal(RobWMHFRolloutPro):
    def __init__(self, module, config, world_model_mapping=None):
        super().__init__(module, config, world_model_mapping)
        self.robot = None
        # 真机通常 batch_size=1，串行执行
        self.real_batch_size = config.get("real_batch_size", 1)

    def _init_robot(self):
        if self.robot is None:
            self.robot = FrankaInterface(self.config)
            self.robot.connect()

    def _close_robot(self):
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    def _generate_minibatch_real(self, prompts) -> DataProto:
        """
        真机专用 rollout 生成逻辑。
        由于真机无法并行，我们循环执行直到凑满 batch_size。
        """
        self._init_robot()
        self.module.eval()

        meta_info = prompts.meta_info
        n_samples = meta_info.get("n_samples", 1)
        # 真机模式下，我们忽略 prompt 中的多任务 batch，每次只执行一个任务
        # 或者假设 prompts 已经是单任务重复多次
        batch_size = prompts.batch.batch_size[0]

        # 为了安全，真机 rollout 建议 batch_size=1，然后在 trainer 层做 gradient_accumulation
        # 这里我们强制按单环境串行生成
        actual_batch_size = 1
        if batch_size > 1:
            print(
                f"[RealRollout] Warning: Real robot batch_size {batch_size} > 1. Sequential execution enabled."
            )

        vla_history = []
        task_records = []
        video_records = []

        # 串行执行直到凑满需要的样本数 (如果 trainer 强要求 batch_size)
        # 建议 trainer 配置 real_batch_size=1
        for idx in range(batch_size):
            # 1. 重置环境
            obs = self.robot.reset()
            task_description = prompts.non_tensor_batch["task_descriptions"][idx]

            step = 0
            max_steps = self.max_steps.get(self.config.task_suite_name, 200)
            active = True
            complete = False

            # 记录该条轨迹的数据
            traj_images = []
            traj_actions = []
            traj_responses = []
            traj_input_ids = []
            traj_attention_masks = []
            traj_pixel_values = []

            while step < max_steps and active:
                # 2. 构造 VLA 输入 (复用父类方法)
                # 需将 obs 转换为 process_input 需要的格式
                input_dict = self._obs_to_input(obs, is_robotwin=False)
                # 构造 batch=1 的输入
                vla_input = self.process_input([input_dict], [task_description])
                vla_input.update(meta_info)

                # 3. VLA 推理
                with torch.no_grad():
                    vla_output = self._generate_one_step(vla_input)

                actions = vla_output["action"]  # [1, chunk, dim]
                action_np = actions[0].cpu().numpy()

                # 记录数据
                traj_actions.append(action_np)
                traj_responses.append(vla_output["responses"])
                traj_input_ids.append(vla_output["input_ids"])
                traj_attention_masks.append(vla_output["attention_mask"])
                traj_pixel_values.append(vla_output["pixel_values"])

                # 4. 真机执行
                next_obs, reward, done, info = self.robot.step(action_np)

                # 记录图像
                img = obs["agentview_image"]
                traj_images.append(img)

                obs = next_obs
                active = not done
                complete = done
                step += self.config.action_chunks_len

            # 保存轨迹记录
            task_records.append(
                {
                    "active": False,
                    "complete": complete,
                    "finish_step": step,
                    "task_file_name": f"real_robot_trial_{idx}",
                }
            )
            video_records.append(
                {"wm_images": traj_images, "env_images": traj_images, "env_dones": []}
            )

            # 将单条轨迹数据堆叠为 Tensor (模拟多 batch 维度)
            # 注意：这里需要将 list 转为 tensor 并 unsqueeze(0) 以匹配 batch 维度
            # 由于是串行，我们需要在循环外 concat，或者这里直接构建好 batch=1 的 tensor
            # 为了适配父类 _prepare_output_batch，我们暂时存储为 list，最后统一处理
            # 这里简化处理：每条轨迹单独生成一个 DataProto 然后 concat
            single_history = []
            for t_idx in range(len(traj_actions)):
                single_history.append(
                    {
                        "responses": traj_responses[t_idx],
                        "input_ids": traj_input_ids[t_idx],
                        "attention_mask": traj_attention_masks[t_idx],
                        "pixel_values": traj_pixel_values[t_idx],
                        "action": traj_actions[t_idx],
                        "step": t_idx,
                        "is_dummy": torch.zeros(
                            1, dtype=torch.bool
                        ),  # 真机数据非 dummy
                    }
                )

            # 生成单条 DataProto
            single_proto = self._prepare_output_batch_evolving(
                prompts.slice(torch.tensor([idx])),  # 切片对应 prompt
                single_history,
                [task_records[-1]],
                [task_description],
                [video_records[-1]],
                batch_size=1,
                max_steps=max_steps,
            )

            if idx == 0:
                final_proto = single_proto
            else:
                final_proto = DataProto.concat([final_proto, single_proto])

        self._close_robot()
        return final_proto

    def generate_sequences(self, prompts) -> DataProto:
        """
        重写生成入口，根据配置选择真机还是仿真。
        """
        if self.config.get("use_real_robot", False):
            print("[Rollout] Using REAL ROBOT mode.")
            return self._generate_minibatch_real(prompts)
        else:
            #  fallback to parent implementation (Sim/WM)
            return super().generate_sequences(prompts)
