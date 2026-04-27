# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Optional

import torch.nn as nn

from ...data import LengthValueDataCollator, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import fix_valuehead_checkpoint
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeLengthMetrics
from .trainer import LengthValueTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


logger = get_logger(__name__)


class MLP2SiLUValueHead(nn.Module):
    """Two-layer value head: Linear(H→H) → SiLU → Linear(H→1).

    Keeps ``self.summary`` as the final ``nn.Linear`` so that TRL's
    ``self.v_head.summary.weight`` device check still works.
    """

    def __init__(self, config, summary_dropout_prob=None):
        super().__init__()
        hidden_size = config.hidden_size
        if summary_dropout_prob is None:
            summary_dropout_prob = getattr(config, "summary_dropout_prob", 0.0)
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.summary = nn.Linear(hidden_size, 1)  # TRL accesses summary.weight
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        output = self.fc(output)
        output = self.act(output)
        output = self.summary(output)
        return output


def run_lenvm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="lenvm", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)

    # Replace TRL's single-Linear value head with 2-layer MLP + SiLU
    dropout_prob = getattr(model_args, "valuehead_dropout", None)
    vhead_param = next(model.v_head.parameters())
    model.v_head = MLP2SiLUValueHead(
        model.pretrained_model.config, summary_dropout_prob=dropout_prob
    ).to(device=vhead_param.device, dtype=vhead_param.dtype)
    logger.info_rank0("Replaced value head with MLP2SiLUValueHead (Linear→SiLU→Linear).")

    data_collator = LengthValueDataCollator(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module
    )

    # Initialize our Trainer
    trainer = LengthValueTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeLengthMetrics(),
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, getattr(training_args, "save_safetensors", True))

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_mae", f"eval_{key}_rmse"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_mae", "eval_rmse"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
