import torch
import numpy as np
import math

from typing import List, Tuple, Dict, Optional, Callable, Union, Any, Iterable
from typing_extensions import ParamSpec, Self, TypeAlias
from torch import Tensor


ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]


class ALSO(torch.optim.Optimizer):
    def __init__(
        self, 
        params: ParamsT,
        n_groups: int, 
        loss_scale: Optional[float] = None, 
        batch_size: Optional[int] = None,
        mode: str = 'optimistic',
        alpha: float = 1.0,
        pi_reg: Optional[Tensor] = None,
        pi_init: Optional[Tensor] = None,
        betas: Tuple[float, float] = (0.9, 0.999), 
        lr: float = 1e-3, 
        weight_decay: float = 1e-3,
        pi_lr: float = 1e-3, 
        pi_decay: float = 1e-2, 
        eps: float = np.finfo(np.float32).eps, 
        amsgrad: bool = False,
    ):
        """
        Initialize the Adaptive Loss Scaling Optimizer from the paper https://arxiv.org/2508.16734.

        Args:
            params: Iterable of parameters to optimize
            n_groups: number of groups used (i.e. c from the paper). Use n_groups=n_objects to have object-level weightening
            loss_scale: loss scaling coefficient fro closure. Please see `example.ipynb` for details
            batch_size: Size of mini-batches used for training
            mode: Optimization mode ('optimistic' or 'descent-ascent')
            alpha: negative momentum from ALSO (used for 'optimistic' mode)
            pi_reg: regularizer for pi (i.e. hat{pi} from the paper). If None, then uniform distribution is used
            pi_init: initial value for pi. If None, then the pi_reg is used
            betas: Adam's betas
            lr: Learning rate for model parameters
            weight_decay: Weight decay coefficient
            pi_lr: Learning rate for pi
            pi_decay: Regularization coefficient for pi
            eps: Small constant for numerical stability
            amsgrad: Whether to use the AMSGrad variant of Adam
        """

        if mode not in ['optimistic', 'descent-ascent']:
            raise ValueError(f"Mode must be 'optimistic' or 'descent-ascent', got {mode}")
        
        if loss_scale is None and batch_size is None:
            raise ValueError(f"At least one of the `loss_scale`, `batch_size` parameters should be not None")
        
        defaults = dict(betas=betas, amsgrad=amsgrad)
        super(ALSO, self).__init__(params, defaults=defaults)

        self.mode = mode
        self.alpha = alpha

        # theta related parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

        # pi related parameters
        self.pi_lr = pi_lr
        self.pi_decay = pi_decay

        # other parameters
        self.eps = eps
        self.loss_scale = n_groups / batch_size if loss_scale is None else loss_scale

        # Initialize pi (weighting distribution)
        self._initialize_pi(n_groups, pi_init, pi_reg)

        # Required momentum terms initialization
        self.__previous_params = None
        self.__pi_intermediate = None
        self.__prev_grads_pi = None
        self.__prev_grads_theta = []
    
    def _initialize_pi(self, n_groups: int, pi_init: Optional[Tensor], pi_reg: Optional[Tensor]) -> None:
        """Initialize the pi distribution and its regularizer."""

        if pi_init is None:
            self.pi = torch.ones(n_groups, requires_grad=False)
        else:
            self.pi = pi_init.detach().clone()
        self.pi = self.pi / self.pi.sum()

        if pi_reg is None:
            self.pi_reg = torch.tensor(1. / n_groups, requires_grad=False)
        else:
            self.pi_reg = pi_reg.detach().clone() / pi_reg.sum()


    def _move_pi_to_device(self, device):
        self.pi = self.pi.to(device)
        self.pi_reg = self.pi_reg.to(device)
        if self.__pi_intermediate is not None:
            self.__pi_intermediate = self.__pi_intermediate.to(device)
            
    def _compute_adam_step(self, p: Tensor, group: Dict[str, Any]) -> Optional[Tensor]:
        """Compute Adam gradient update for a parameter."""

        if p.grad is None:
            return None

        grad = p.grad.data
        amsgrad = group['amsgrad']
        state = self.state[p]
        
        # State initialization
        if f'step' not in state:
            state[f'step'] = 0
            state[f'exp_avg'] = torch.zeros_like(p.data)
            state[f'exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                state[f'max_exp_avg_sq'] = torch.zeros_like(p.data)
        
        exp_avg, exp_avg_sq = state[f'exp_avg'], state[f'exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state[f'max_exp_avg_sq']
        beta1, beta2 = group['betas']
        
        state[f'step'] += 1
        
        if self.weight_decay != 0:
            grad = grad.add(p.data, alpha=self.weight_decay)
        
        # Update momentum
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().add_(self.eps)
        else:
            denom = exp_avg_sq.sqrt().add_(self.eps)

        # Bias correction
        bias_correction1 = 1 - beta1 ** state[f'step']
        bias_correction2 = 1 - beta2 ** state[f'step']
        step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1
        
        return -step_size * exp_avg / denom

    def _update_pi(self, pi_grad: Tensor) -> Tensor:
        """Update the pi values using the current gradient."""

        pi_new_log = (torch.log(self.pi + self.eps) + 
                      self.pi_decay * self.pi_lr * torch.log(self.pi_reg) - 
                      self.pi_lr * pi_grad)
        pi_new_log /= 1 + self.pi_decay * self.pi_lr
        return torch.nn.functional.softmax(pi_new_log, dim=-1)

    def _descent_ascent_step(self, closure, groups_indexes) -> float:
        """Perform a descent-ascent optimization step."""

        # Compute gradients using pi
        pi_selected = self.pi[groups_indexes]
        losses, _ = closure(pi_selected, self.loss_scale)
        loss = losses.mean().item()

        # Update parameters
        for group in self.param_groups: 
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(self._compute_adam_step(p, group))

        # Update pi
        pi_grad_compressed = -losses.clone().detach()
        pi_grad = torch.zeros_like(self.pi, requires_grad=False)
        pi_grad.index_add_(0, groups_indexes.flatten(), pi_grad_compressed.flatten())
        self.pi = self._update_pi(pi_grad)
        return loss

    def _optimistic_intermediate_step(self):
        """Perform the intermediate step in optimistic mode."""

        self.__previous_params = []
        # Make step over model parameters using old gradient
        param_idx = 0
        for group in self.param_groups: 
            for p in group['params']: 
                self.__previous_params.append(p.data)

                if param_idx < len(self.__prev_grads_theta) and self.__prev_grads_theta[param_idx] is not None:
                    p.grad = self.alpha * self.__prev_grads_theta[param_idx]
                    p.data = p.data + self._compute_adam_step(p, group)
                param_idx += 1

        # Make step over pi using previous gradient
        if self.__prev_grads_pi is not None:
            self.__pi_intermediate = self._update_pi(self.__prev_grads_pi)
        else:
            self.__pi_intermediate = self.pi.clone()

    def _optimistic_main_step(self, closure, groups_indexes) -> float:
        """Perform the main step in optimistic mode."""

        # Compute gradients using intermediate pi
        pi_selected = self.__pi_intermediate[groups_indexes]
        losses, _ = closure(pi_selected, self.loss_scale)
        loss = losses.mean().item()

        # Save gradients and update parameters
        self.__prev_grads_theta = []
        param_idx = 0
        for group in self.param_groups: 
            for p in group['params']:
                if p.grad is not None:
                    self.__prev_grads_theta.append(p.grad.data)
                    p.data.copy_(self.__previous_params[param_idx])
                    p.data.add_(self._compute_adam_step(p, group))
                else:
                    self.__prev_grads_theta.append(None)
                param_idx += 1

        # Update pi
        pi_grad_compressed = -losses.clone().detach()
        pi_grad = torch.zeros_like(self.pi, requires_grad=False)
        pi_grad.index_add_(0, groups_indexes.flatten(), pi_grad_compressed.flatten())
        self.__prev_grads_pi = pi_grad
        self.pi = self._update_pi(pi_grad)
        return loss

    def step(self, closure, groups_indexes) -> float:
        """
        Perform a single optimization step.
        
        Args:
            closure: A callable that takes pi_selected and loss_scale, returns losses and other outputs
            groups_indexes: Tensor of group indices for the current batch
            
        Returns:
            float: The mean loss value
        """
        # Get device from the closure
        if hasattr(closure, 'device'):
            self._move_pi_to_device(closure.device)

            # Ensure groups_indexes is a tensor
            if not isinstance(groups_indexes, torch.Tensor):
                groups_indexes = torch.tensor(groups_indexes, device=closure.device)

        if self.mode == 'optimistic':
            self._optimistic_intermediate_step()
            return self._optimistic_main_step(closure, groups_indexes)
        elif self.mode == 'descent-ascent':
            return self._descent_ascent_step(closure, groups_indexes)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
