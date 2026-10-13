"""Experimental fixed-weight foreground/background joint OT replacement.

Copy to matcher/SQAdapter.py in an experimental checkout to use. No target
labels are required. See README.md for the changed model and numerical limits.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Adapter(nn.Module):
    def __init__(self, input_dim, args, device):
        super().__init__()
        hidden = int(input_dim * args.adapter_rate)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(args.adapter_dropout), nn.Linear(hidden, input_dim),
        ).to(device)
        nn.init.zeros_(self.adapter[4].weight)
        nn.init.zeros_(self.adapter[4].bias)

    def forward(self, x):
        return x + self.adapter(x)


class SQAdapter:
    def __init__(self, ref_masks_pool, nshot, device, args=None, logger=None):
        self.ref_masks_pool, self.nshot = ref_masks_pool, nshot
        self.device, self.args, self.logger = device, args, logger
        self.epsilon = 0.001  # Fixed model constant; intentionally not read from YAML.
        self.max_iter = args.ot_max_iter
        self.tol = args.ot_marginal_tol
        self.inner_steps = args.ot_inner_steps
        self.backtracks = args.ot_backtracking_steps
        self.weights = (
            float(args.EMD_weight), float(args.proto_weight),
            float(args.s2q2s_prior_map_weight), float(args.prior_distill_weight),
        )
        self.bg_weight = args.prior_distill_bg_weight

    @staticmethod
    def cosine_similarity(x, y):
        return F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).T

    def response(self, positive, negative, locations):
        # One definition for teacher, student and cycle: cosine difference,
        # ReLU, then per-row min-max. Keep extrema in the differentiation graph.
        margin = (self.cosine_similarity(positive, locations)
                  - self.cosine_similarity(negative, locations))
        positive_margin = F.relu(margin)
        low = positive_margin.amin(dim=-1, keepdim=True)
        high = positive_margin.amax(dim=-1, keepdim=True)
        return (positive_margin - low) / (high - low + 1e-8)

    @staticmethod
    @torch.no_grad()
    def marginals(teacher, foreground):
        target_mass = teacher.clamp_min(1e-2)  # Original EMD prior floor.
        nu = target_mass / target_mass.sum()
        alpha = (nu * teacher).sum()
        mu = torch.where(foreground, alpha / foreground.sum(),
                         (1 - alpha) / (~foreground).sum())
        return mu.detach(), nu.detach(), alpha.detach()

    def optimizer_scheduler(self, adapter, steps, lr):
        optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
        warmup = int(steps * 0.1)

        def schedule(step):
            if step < warmup:
                return float(step) / warmup
            progress = float(step - warmup) / (steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    @staticmethod
    def marginal_error(pi, mu, nu):
        # Stopping criterion for the two transport marginals.
        return torch.maximum(((pi.sum(1) - mu) / mu).abs().max(),
                             ((pi.sum(0) - nu) / nu).abs().max())

    @torch.no_grad()
    def sinkhorn(self, cost, mu, nu):
        # Float64 here limits cancellation in corrected costs; the adapter stays
        # float32. Stop on BOTH marginals, not on change of dual potentials.
        log_kernel = -cost.double() / self.epsilon
        log_mu, log_nu = mu.double().log(), nu.double().log()
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for step in range(self.max_iter):
            u = log_mu - torch.logsumexp(log_kernel + v[None, :], dim=1)
            v = log_nu - torch.logsumexp(log_kernel + u[:, None], dim=0)
            if (step + 1) % 10 == 0 or step == self.max_iter - 1:
                pi = (log_kernel + u[:, None] + v[None, :]).exp().to(cost.dtype)
                if self.marginal_error(pi, mu, nu) <= self.tol:
                    return pi
        return (log_kernel + u[:, None] + v[None, :]).exp().to(cost.dtype)

    @staticmethod
    def bce(prediction, teacher):
        return F.binary_cross_entropy(prediction.clamp(1e-6, 1 - 1e-6),
                                      teacher, reduction='none')

    def losses(self, reference, target, foreground, anchor, teacher, mu, nu, pi):
        cost = 1 - self.cosine_similarity(reference, target)
        entropy = (pi * (pi.clamp_min(1e-30).log() - 1)).sum()
        emd = (pi * cost).sum() + self.epsilon * entropy

        foreground_mass = pi[foreground].sum(0)
        transported_prior = foreground_mass / nu
        source_proto = reference[foreground].mean(0, keepdim=True)
        target_proto = (foreground_mass @ target).unsqueeze(0)
        proto = (1 - self.cosine_similarity(source_proto, anchor).squeeze()
                 + 1 - self.cosine_similarity(source_proto, target_proto).squeeze())

        soft_target = (pi[foreground] / mu[foreground, None]) @ target
        background_proto = reference[~foreground].mean(0, keepdim=True)
        cycle_prior = self.response(soft_target, background_proto, reference)
        mask = foreground.to(reference.dtype)[None, :].expand_as(cycle_prior)
        cycle_per_row = self.bce(cycle_prior, mask).mean(1)
        cycle = (mu[foreground] * cycle_per_row).sum() / mu[foreground].sum()

        student_prior = self.response(source_proto, background_proto, target).squeeze(0)
        # Soft class weighting replaces top/bottom-K. Keep the old 1+beta total
        # scale, rather than silently normalizing it to one.
        positive_weights = teacher / teacher.sum()
        negative_weights = (1 - teacher) / (1 - teacher).sum()
        confidence_weights = positive_weights + self.bg_weight * negative_weights
        transported_dist = (confidence_weights * self.bce(transported_prior, teacher)).sum()
        student_dist = (confidence_weights * self.bce(student_prior, teacher)).sum()
        dist = 0.5 * (transported_dist + student_dist)
        return emd, proto, cycle, dist

    def objective(self, losses):
        return sum(weight * loss for weight, loss in zip(self.weights, losses))

    def solve_plan(self, reference, target, foreground, anchor, teacher, mu, nu, initial=None):
        # Joint/block optimization: these tensors are constants during Pi updates.
        reference, target = reference.detach(), target.detach()
        anchor, teacher = anchor.detach(), teacher.detach()
        cost = 1 - self.cosine_similarity(reference, target)
        pi = self.sinkhorn(cost, mu, nu) if initial is None else initial.detach()
        if not torch.isfinite(pi).all():
            return None
        for _ in range(self.inner_steps):
            with torch.enable_grad():
                variable = pi.detach().requires_grad_(True)
                terms = self.losses(reference, target, foreground, anchor, teacher, mu, nu, variable)
                semantic = sum(w * term for w, term in zip(self.weights[1:], terms[1:]))
                gradient, = torch.autograd.grad(semantic, variable)
            # Entropy remains in Sinkhorn's subproblem; do NOT add its gradient
            # again to the corrected linear cost.
            corrected_cost = cost + gradient.detach() / self.weights[0]
            if not torch.isfinite(corrected_cost).all():
                break
            with torch.no_grad():
                old_value = self.objective(terms).item()
                proposal = self.sinkhorn(corrected_cost, mu, nu)
                if not torch.isfinite(proposal).all():
                    break
                for backtrack in range(self.backtracks):
                    fraction = 0.5 ** backtrack
                    candidate = (1 - fraction) * pi + fraction * proposal
                    value = self.objective(self.losses(
                        reference, target, foreground, anchor, teacher, mu, nu, candidate))
                    if value.item() < old_value:
                        pi = candidate
                        break
        return pi.detach()

    def transform(self, ref_feats, tar_feat, emd_prior_map=None, prior_distill_prior_map=None):
        # Support callers wrapped in torch.no_grad(); do not mutate input features.
        reference, target = ref_feats.detach().float(), tar_feat.detach().float()
        masks = self.ref_masks_pool.to(reference.device).reshape(-1)
        external_teacher = (prior_distill_prior_map if prior_distill_prior_map is not None
                            else emd_prior_map)
        if external_teacher is not None:
            external_teacher = external_teacher.detach().to(target).reshape(-1)
        output = reference.clone()
        length = reference.shape[0] // self.nshot
        for shot in range(self.nshot):
            start, end = shot * length, (shot + 1) * length
            original, foreground = reference[start:end], masks[start:end] == 1
            # Both prototypes and both class-conditional transports must exist.
            if not foreground.any() or foreground.all():
                continue
            with torch.no_grad():
                anchor = original[foreground].mean(0, keepdim=True)
                background = original[~foreground].mean(0, keepdim=True)
                teacher = (self.response(anchor, background, target).squeeze(0)
                           if external_teacher is None else external_teacher)
                teacher = teacher.detach()
                # Zero class mass makes Pi_fg / mu_fg and distillation weights
                # undefined. Keep this shot unchanged instead of inventing mass.
                if (not torch.isfinite(teacher).all()
                        or teacher.sum() <= 0 or (1 - teacher).sum() <= 0):
                    continue
                mu, nu, _ = self.marginals(teacher, foreground)
                if not torch.isfinite(mu).all() or not (mu > 0).all():
                    continue
            mlp = nn.Sequential(*[Adapter(original.shape[1], self.args, original.device)
                                  for _ in range(self.args.adapter_num)])
            mlp.train()
            optimizer, scheduler = self.optimizer_scheduler(
                mlp, self.args.finetune_epochs, self.args.adapter_lr)
            pi = None
            for _ in range(self.args.finetune_epochs):
                with torch.enable_grad():
                    adapted = original.clone()
                    adapted[foreground] = mlp(original[foreground])
                    # Reuse this forward pass: no new dropout sample between blocks.
                    pi = self.solve_plan(adapted, target, foreground, anchor, teacher, mu, nu, pi)
                    if pi is None:
                        break
                    terms = self.losses(adapted, target, foreground, anchor, teacher, mu, nu, pi)
                    total = self.objective(terms)
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    optimizer.step()
                    scheduler.step()
            mlp.eval()
            with torch.no_grad():
                result = original.clone()
                result[foreground] = mlp(original[foreground])
                output[start:end] = result
        if self.args.postnorm:
            output = F.normalize(output, dim=1)
        return output.to(ref_feats.dtype).detach()
