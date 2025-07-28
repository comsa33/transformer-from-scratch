"""
Transformer 모델 학습을 위한 Trainer 클래스

학습 루프, 검증, 체크포인팅 등을 관리합니다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, List, Tuple, Union, Callable, Any
import os
import time
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

from .loss import create_transformer_loss
from .optimizer import create_optimizer, create_scheduler


class TrainingConfig:
    """학습 설정을 관리하는 클래스"""
    
    def __init__(self,
                 # 기본 설정
                 output_dir: str = "./checkpoints",
                 num_train_epochs: int = 10,
                 per_device_train_batch_size: int = 32,
                 per_device_eval_batch_size: int = 64,
                 gradient_accumulation_steps: int = 1,
                 
                 # Optimizer 설정
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 adam_beta1: float = 0.9,
                 adam_beta2: float = 0.999,
                 adam_epsilon: float = 1e-8,
                 max_grad_norm: float = 1.0,
                 
                 # Scheduler 설정
                 lr_scheduler_type: str = "cosine",
                 warmup_steps: int = 1000,
                 warmup_ratio: Optional[float] = None,
                 
                 # 학습 전략
                 fp16: bool = False,
                 gradient_checkpointing: bool = False,
                 
                 # 로깅 및 저장
                 logging_steps: int = 100,
                 eval_steps: int = 500,
                 save_steps: int = 1000,
                 save_total_limit: int = 3,
                 metric_for_best_model: str = "loss",
                 greater_is_better: bool = False,
                 
                 # 기타
                 seed: int = 42,
                 dataloader_num_workers: int = 4,
                 remove_unused_columns: bool = True,
                 label_names: Optional[List[str]] = None,
                 load_best_model_at_end: bool = True,
                 resume_from_checkpoint: Optional[str] = None,
                 **kwargs):
        
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        
        self.fp16 = fp16
        self.gradient_checkpointing = gradient_checkpointing
        
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        
        self.seed = seed
        self.dataloader_num_workers = dataloader_num_workers
        self.remove_unused_columns = remove_unused_columns
        self.label_names = label_names or ["labels"]
        self.load_best_model_at_end = load_best_model_at_end
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # 추가 kwargs 저장
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict:
        """설정을 딕셔너리로 변환"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save_to_json(self, path: str):
        """설정을 JSON 파일로 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, path: str):
        """JSON 파일에서 설정 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class TrainerState:
    """학습 상태를 추적하는 클래스"""
    
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.total_flos = 0
        self.log_history = []
        self.best_metric = None
        self.best_model_checkpoint = None
    
    def save_to_json(self, path: str):
        """상태를 JSON 파일로 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, path: str):
        """JSON 파일에서 상태 로드"""
        state = cls()
        with open(path, 'r', encoding='utf-8') as f:
            state.__dict__.update(json.load(f))
        return state


class Trainer:
    """Transformer 모델 학습을 위한 Trainer 클래스"""
    
    def __init__(self,
                 model: nn.Module,
                 args: TrainingConfig,
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 eval_dataset: Optional[torch.utils.data.Dataset] = None,
                 data_collator: Optional[Callable] = None,
                 tokenizer: Optional[Any] = None,
                 compute_metrics: Optional[Callable] = None,
                 callbacks: Optional[List[Callable]] = None,
                 optimizers: Tuple[Optional[torch.optim.Optimizer], 
                                 Optional[torch.optim.lr_scheduler._LRScheduler]] = (None, None)):
        
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        
        # Optimizer와 Scheduler
        self.optimizer, self.lr_scheduler = optimizers
        
        # 상태 초기화
        self.state = TrainerState()
        
        # 로거 설정
        self.logger = self._setup_logger()
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Mixed precision 설정
        self.scaler = GradScaler() if args.fp16 else None
        
        # Loss function
        self.loss_fn = None
        
        # 시드 설정
        self._set_seed(args.seed)
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Output directory 생성 (로그 파일을 위해)
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # 파일 핸들러
        log_file = os.path.join(self.args.output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _set_seed(self, seed: int):
        """랜덤 시드 설정"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Optimizer와 Scheduler 생성"""
        if self.optimizer is None:
            self.optimizer = create_optimizer(
                self.model,
                optimizer_type="adamw",
                learning_rate=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon
            )
        
        if self.lr_scheduler is None:
            # Warmup steps 계산
            if self.args.warmup_ratio is not None:
                warmup_steps = int(num_training_steps * self.args.warmup_ratio)
            else:
                warmup_steps = self.args.warmup_steps
            
            self.lr_scheduler = create_scheduler(
                self.optimizer,
                scheduler_type=self.args.lr_scheduler_type,
                warmup_steps=warmup_steps,
                max_steps=num_training_steps
            )
    
    def get_train_dataloader(self) -> DataLoader:
        """학습 DataLoader 생성"""
        if self.train_dataset is None:
            raise ValueError("Training dataset is not provided")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
            pin_memory=True
        )
    
    def get_eval_dataloader(self) -> DataLoader:
        """평가 DataLoader 생성"""
        if self.eval_dataset is None:
            return None
        
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
            pin_memory=True
        )
    
    def compute_loss(self, model: nn.Module, inputs: Dict) -> torch.Tensor:
        """Loss 계산"""
        labels = inputs.pop("labels", None)
        
        # Transformer 모델의 forward 인터페이스에 맞게 변환
        if 'src_ids' in inputs and 'tgt_ids' in inputs:
            outputs = model(
                src_ids=inputs['src_ids'],
                tgt_ids=inputs['tgt_ids'],
                src_mask=inputs.get('src_mask'),
                tgt_mask=inputs.get('tgt_mask')
            )
        else:
            outputs = model(**inputs)
        
        if self.loss_fn is None:
            # 기본 loss function 설정
            self.loss_fn = create_transformer_loss(
                task_type='generation',
                vocab_size=model.tgt_vocab_size if hasattr(model, 'tgt_vocab_size') else 30000,
                label_smoothing=0.1
            )
        
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs
        
        if labels is not None:
            loss, metrics = self.loss_fn(logits, labels)
            return loss
        else:
            return outputs
    
    def training_step(self, model: nn.Module, inputs: Dict) -> torch.Tensor:
        """단일 학습 스텝"""
        model.train()
        
        if self.args.fp16:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        if self.args.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.detach()
    
    def train(self):
        """전체 학습 루프"""
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()
        
        # 총 학습 스텝 계산
        total_train_batch_size = (
            self.args.per_device_train_batch_size * 
            self.args.gradient_accumulation_steps
        )
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * self.args.num_train_epochs
        
        # Optimizer와 Scheduler 생성
        self.create_optimizer_and_scheduler(num_training_steps)
        
        # 체크포인트에서 재개
        if self.args.resume_from_checkpoint:
            self._load_checkpoint(self.args.resume_from_checkpoint)
        
        # 학습 시작
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        self.logger.info(f"  Total train batch size = {total_train_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {num_training_steps}")
        
        self.state.global_step = 0
        tr_loss = torch.tensor(0.0).to(self.device)
        
        self._call_callback("on_train_begin")
        
        for epoch in range(self.args.num_train_epochs):
            self.state.epoch = epoch
            self._call_callback("on_epoch_begin")
            
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
            
            for step, inputs in enumerate(epoch_iterator):
                # 입력을 디바이스로 이동
                inputs = self._prepare_inputs(inputs)
                
                # 학습 스텝
                tr_loss_step = self.training_step(self.model, inputs)
                tr_loss += tr_loss_step
                
                # Gradient accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.args.max_grad_norm
                    )
                    
                    # Optimizer step
                    if self.args.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.state.global_step += 1
                    
                    # 로깅
                    if self.state.global_step % self.args.logging_steps == 0:
                        self._log(tr_loss)
                        tr_loss = torch.tensor(0.0).to(self.device)
                    
                    # 평가
                    if (self.args.eval_steps > 0 and 
                        self.state.global_step % self.args.eval_steps == 0):
                        self.evaluate()
                    
                    # 저장
                    if self.state.global_step % self.args.save_steps == 0:
                        self._save_checkpoint()
                
                epoch_iterator.set_description(
                    f"Epoch {epoch} - Loss: {tr_loss_step.item():.4f}"
                )
            
            self._call_callback("on_epoch_end")
        
        self._call_callback("on_train_end")
        
        # 최고 성능 모델 로드
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint:
            if os.path.exists(self.state.best_model_checkpoint):
                self._load_checkpoint(self.state.best_model_checkpoint)
            else:
                self.logger.warning(f"Best model checkpoint not found: {self.state.best_model_checkpoint}")
        
        return self.state.log_history
    
    def evaluate(self, eval_dataset: Optional[torch.utils.data.Dataset] = None) -> Dict:
        """모델 평가"""
        eval_dataloader = self.get_eval_dataloader() if eval_dataset is None else \
                         DataLoader(eval_dataset, batch_size=self.args.per_device_eval_batch_size)
        
        if eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_eval_loss = 0.0
        total_eval_steps = 0
        
        all_preds = []
        all_labels = []
        
        self._call_callback("on_evaluate")
        
        with torch.no_grad():
            for inputs in tqdm(eval_dataloader, desc="Evaluating"):
                inputs = self._prepare_inputs(inputs)
                
                if self.args.fp16:
                    with autocast():
                        loss = self.compute_loss(self.model, inputs)
                else:
                    loss = self.compute_loss(self.model, inputs)
                
                total_eval_loss += loss.item()
                total_eval_steps += 1
                
                # 예측값과 레이블 수집 (metrics 계산용)
                if self.compute_metrics is not None:
                    # Transformer 모델의 forward 인터페이스에 맞게 변환
                    if 'src_ids' in inputs and 'tgt_ids' in inputs:
                        outputs = self.model(
                            src_ids=inputs['src_ids'],
                            tgt_ids=inputs['tgt_ids'],
                            src_mask=inputs.get('src_mask'),
                            tgt_mask=inputs.get('tgt_mask')
                        )
                    else:
                        outputs = self.model(**inputs)
                    
                    if isinstance(outputs, dict):
                        logits = outputs["logits"]
                    else:
                        logits = outputs
                    
                    preds = logits.argmax(dim=-1)
                    all_preds.append(preds.cpu())
                    all_labels.append(inputs["labels"].cpu())
        
        # 평균 loss
        eval_loss = total_eval_loss / total_eval_steps
        
        # 추가 metrics 계산
        eval_metrics = {"eval_loss": eval_loss}
        
        if self.compute_metrics is not None and len(all_preds) > 0:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            metrics = self.compute_metrics((all_preds, all_labels))
            eval_metrics.update(metrics)
        
        # 최고 성능 모델 추적
        metric_key = self.args.metric_for_best_model
        if metric_key == 'loss':
            metric_key = 'eval_loss'
        
        if self.state.best_metric is None or self._is_better(eval_metrics):
            self.state.best_metric = eval_metrics[metric_key]
            self.state.best_model_checkpoint = self._get_checkpoint_path()
        
        # 로그에 추가
        self.state.log_history.append({
            **eval_metrics,
            "epoch": self.state.epoch,
            "step": self.state.global_step
        })
        
        self.logger.info(f"Evaluation results: {eval_metrics}")
        
        return eval_metrics
    
    def _prepare_inputs(self, inputs: Dict) -> Dict:
        """입력을 디바이스로 이동"""
        prepared = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.device)
            else:
                prepared[k] = v
        return prepared
    
    def _log(self, loss: torch.Tensor):
        """학습 로그 기록"""
        lr = self.lr_scheduler.get_last_lr()[0]
        avg_loss = loss.item() / self.args.logging_steps
        
        log_info = {
            "loss": avg_loss,
            "learning_rate": lr,
            "epoch": self.state.epoch,
            "step": self.state.global_step
        }
        
        self.state.log_history.append(log_info)
        self.logger.info(
            f"Step {self.state.global_step} - Loss: {avg_loss:.4f}, LR: {lr:.6f}"
        )
    
    def _save_checkpoint(self):
        """체크포인트 저장"""
        checkpoint_path = self._get_checkpoint_path()
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 모델 저장
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_path, "pytorch_model.bin")
        )
        
        # Optimizer 상태 저장
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_path, "optimizer.pt")
        )
        torch.save(
            self.lr_scheduler.state_dict(),
            os.path.join(checkpoint_path, "scheduler.pt")
        )
        
        # 학습 상태 저장
        self.state.save_to_json(
            os.path.join(checkpoint_path, "trainer_state.json")
        )
        
        # 설정 저장
        self.args.save_to_json(
            os.path.join(checkpoint_path, "training_args.json")
        )
        
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")
        
        # 오래된 체크포인트 삭제
        self._rotate_checkpoints()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        # 모델 로드
        self.model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
        )
        
        # Optimizer 상태 로드
        if os.path.exists(os.path.join(checkpoint_path, "optimizer.pt")):
            self.optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint_path, "optimizer.pt"))
            )
        
        if os.path.exists(os.path.join(checkpoint_path, "scheduler.pt")):
            self.lr_scheduler.load_state_dict(
                torch.load(os.path.join(checkpoint_path, "scheduler.pt"))
            )
        
        # 학습 상태 로드
        if os.path.exists(os.path.join(checkpoint_path, "trainer_state.json")):
            self.state = TrainerState.from_json(
                os.path.join(checkpoint_path, "trainer_state.json")
            )
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _get_checkpoint_path(self) -> str:
        """체크포인트 경로 생성"""
        return os.path.join(
            self.args.output_dir,
            f"checkpoint-{self.state.global_step}"
        )
    
    def _rotate_checkpoints(self):
        """오래된 체크포인트 삭제"""
        checkpoints = []
        for d in os.listdir(self.args.output_dir):
            if d.startswith("checkpoint-"):
                checkpoints.append(d)
        
        if len(checkpoints) <= self.args.save_total_limit:
            return
        
        # 스텝 번호로 정렬
        checkpoints = sorted(
            checkpoints,
            key=lambda x: int(x.split("-")[1])
        )
        
        # 오래된 체크포인트 삭제
        for checkpoint in checkpoints[:-self.args.save_total_limit]:
            checkpoint_path = os.path.join(self.args.output_dir, checkpoint)
            if checkpoint_path != self.state.best_model_checkpoint:
                import shutil
                shutil.rmtree(checkpoint_path)
                self.logger.info(f"Deleted old checkpoint: {checkpoint_path}")
    
    def _is_better(self, metrics: Dict) -> bool:
        """현재 metrics가 더 나은지 확인"""
        metric_key = self.args.metric_for_best_model
        if metric_key == 'loss':
            metric_key = 'eval_loss'
            
        current = metrics[metric_key]
        best = self.state.best_metric
        
        if self.args.greater_is_better:
            return current > best
        else:
            return current < best
    
    def _call_callback(self, event: str):
        """콜백 호출"""
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(self, self.state, self.model)