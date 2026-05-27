# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset, Sampler
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)

# Extra scalar keys the model's forward may emit alongside `loss`. We
# accumulate them across grad-accum microbatches and inject the running
# mean into HF Trainer's `log()` payload so wandb/tensorboard pick them up.
# Values that are absent on a given step are silently skipped.
_EXTRA_LOSS_KEYS = (
    "action_loss",
    "aux_loss",
    "aux_xy_loss",
    "aux_z_loss",
    "aux_gt_loss",
    "aux_pi3x_loss",
)


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class DualBrainTrainer(transformers.Trainer):
    def __init__(self, **kwargs):
        self.compute_dtype = kwargs.pop("compute_dtype")
        super().__init__(**kwargs)
        # Allowlist numpy globals for safe RNG state unpickling in PyTorch 2.1+
        torch.serialization.add_safe_globals(
            [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
        )
        # Rank-local accumulator for extra scalars emitted by model.forward
        # (action_loss / aux_loss / aux_xy_loss / aux_z_loss). Reset whenever
        # we flush into `log()`. NOTE: these are rank-local averages, not
        # all-reduced like HF's main `loss` — fine for monitoring.
        self._extra_loss_sum: dict = defaultdict(float)
        self._extra_loss_n: int = 0

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]
        # Accumulate optional scalar metrics for `log()`. Skipped silently
        # when the key is absent (e.g. baseline run with no pi3x distill).
        if self.model.training:
            for key in _EXTRA_LOSS_KEYS:
                if key not in outputs:
                    continue
                v = outputs[key]
                if torch.is_tensor(v):
                    v = v.detach().float().item()
                self._extra_loss_sum[key] += float(v)
            self._extra_loss_n += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        # Inject the running means of extra scalars into the same payload HF
        # Trainer sends to its callbacks (wandb, tensorboard, console).
        if self._extra_loss_n > 0:
            for key, total in self._extra_loss_sum.items():
                logs[key] = total / self._extra_loss_n
            self._extra_loss_sum.clear()
            self._extra_loss_n = 0
        return super().log(logs, start_time=start_time)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # Optional separate LR for the LLM backbone. When set (via
            # --llm-learning-rate, stashed on args by gr00t_finetune.py), the LLM
            # params get their own param group(s) at that LR while everything else
            # keeps args.learning_rate. None => single-LR baseline behavior.
            llm_lr = getattr(self.args, "llm_learning_rate", None)

            def _is_llm(name: str) -> bool:
                return "eagle_model.language_model" in name

            def _group(decay: bool, llm: bool) -> dict:
                in_decay = (lambda n: n in decay_parameters) if decay else (lambda n: n not in decay_parameters)
                g = {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if p.requires_grad and in_decay(n) and (_is_llm(n) == llm)
                    ],
                    "weight_decay": self.args.weight_decay if decay else 0.0,
                }
                if llm and llm_lr is not None:
                    g["lr"] = llm_lr
                return g

            if llm_lr is not None and llm_lr > 0:
                optimizer_grouped_parameters = [
                    _group(decay=True, llm=False),
                    _group(decay=False, llm=False),
                    _group(decay=True, llm=True),
                    _group(decay=False, llm=True),
                ]
                # Drop empty groups (e.g. LLM frozen): torch errors only if ALL
                # groups are empty, but empty groups are noise — filter them.
                optimizer_grouped_parameters = [
                    grp for grp in optimizer_grouped_parameters if len(grp["params"]) > 0
                ]
                n_llm = sum(
                    p.numel()
                    for n, p in opt_model.named_parameters()
                    if p.requires_grad and _is_llm(n)
                )
                print(
                    f"[per-group LR] LLM @ lr={llm_lr}, rest @ lr={self.args.learning_rate} "
                    f"(trainable LLM params: {n_llm / 1e6:.1f}M)"
                )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
