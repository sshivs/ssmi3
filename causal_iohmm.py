import argparse
import json
import os
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jax import random
from sklearn.linear_model import LogisticRegression

from advi_ssm_mimic4.advi_hmm import ADVIHMM
from advi_ssm_mimic4.synthetic_data_generator import (
    save_truth_npz,
    simulate_causal_iohmm,
    split_synthetic,
    transition_effect_from_params,
)
from advi_ssm_mimic4.exp_helper import _find_permutation, _generate_global_theta

torch.manual_seed(0)

LOG_EPS = 1e-16


def make_config(result_dir, M, seed=1234, num_iters=100, batch_size=128):
    return SimpleNamespace(
        K_true=2,
        K_train=2,
        D=2,
        D_lstm=2,
        S=0,
        C=2,
        M=M,
        T=24,
        N=256,
        N_train=128,
        N_val=64,
        N_test=64,
        BATCH_SIZE=batch_size,
        lags=1,
        num_iters=num_iters,
        dist="ar",
        discriminator_constraint=False,
        similarity_constraint=False,
        hyperparameters={
            "discriminator_coeff": [1e5],
            "similarity_coeff": [1e11],
            "lr": [5e-3],
            "log_lik_coeff": [1.0],
            "determinants_coeff": [1e2],
            "global_vars_entropy_coeff": [1e2],
            "local_vars_entropy_coeff": [1e3],
            "priors_coeff": [1.0],
            "dirichlet_concentration": [8.0],
        },
        use_recognition_marginals=True,
        sample_global=True,
        sample_local=True,
        sample_feature=True,
        old_ll=False,
        init_true_hmm=False,
        trained_lstm_file="",
        trained_lstm_config="",
        cluster_assignment_file="",
        debug_output_file=os.path.join(result_dir, "debug.log"),
        impute=False,
    )


class PropensityLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        logits = self.head(h).squeeze(-1)
        return logits


class WeightedCausalADVIHMM(ADVIHMM):
    """
    ADVIHMM variant with per-time expected log joint terms for input-driven
    transitions and optional sequential importance weights.

    likelihood_loss:
        - "factorized": uses q(s_t) q(s_{t+1}) from the amortized posterior
        - "exact_pairwise": recomputes HMM gamma/xi with time-varying transitions
          and uses those exact pairwise marginals in the transition term
    """

    def __init__(self, *args, likelihood_loss="exact_pairwise", **kwargs):
        super().__init__(*args, **kwargs)
        self.likelihood_loss = likelihood_loss

    def _build_transition_features(self, seq_inputs, t, ws_dim, lags):
        if seq_inputs is None or ws_dim == 0:
            return jnp.zeros((ws_dim,))

        pieces = []
        m_dim = seq_inputs.shape[-1]
        for lag in range(lags):
            # The edge z_{t-1} -> z_t uses inputs available at t-1.
            idx = t - 1 - lag
            if idx >= 0:
                pieces.append(seq_inputs[idx])
            else:
                pieces.append(jnp.zeros((m_dim,)))
        feat = jnp.concatenate(pieces, axis=0)
        if feat.shape[0] + 1 == ws_dim:
            feat = jnp.concatenate([feat, jnp.ones((1,))], axis=0)
        elif feat.shape[0] < ws_dim:
            feat = jnp.pad(feat, (0, ws_dim - feat.shape[0]))
        elif feat.shape[0] > ws_dim:
            feat = feat[:ws_dim]
        return feat

    def _input_driven_log_transitions(self, seq_inputs, trans_params, T):
        base_log = jnp.log(jnp.clip(trans_params["Ps"], LOG_EPS, 1.0))
        if self.M == 0 or seq_inputs is None or "Ws" not in trans_params or trans_params["Ws"].size == 0:
            return jnp.repeat(base_log[None, :, :], repeats=max(T - 1, 1), axis=0)

        Ws = trans_params["Ws"]
        lags = getattr(self.transitions, "lags", 1)
        logs = []
        for t in range(1, T):
            feat = self._build_transition_features(seq_inputs, t, Ws.shape[-1], lags)
            target_shift = jnp.dot(Ws, feat)
            log_t = base_log + target_shift[None, :]
            log_t = log_t - jax_logsumexp(log_t, axis=1, keepdims=True)
            logs.append(log_t)
        if not logs:
            return jnp.repeat(base_log[None, :, :], repeats=1, axis=0)
        return jnp.stack(logs, axis=0)

    def _forward_backward_pairwise(self, log_pi0, log_trans, log_obs):
        """Return gamma_t(k) and xi_t(k,k') for a single sequence."""
        T, K = log_obs.shape
        alphas = []
        alpha = log_pi0 + log_obs[0]
        alpha = alpha - jax_logsumexp(alpha)
        alphas.append(alpha)
        for t in range(1, T):
            prev = alphas[-1][:, None] + log_trans[t - 1]
            alpha = log_obs[t] + jax_logsumexp(prev, axis=0)
            alpha = alpha - jax_logsumexp(alpha)
            alphas.append(alpha)
        alphas = jnp.stack(alphas, axis=0)

        betas = [jnp.zeros((K,))]
        beta = jnp.zeros((K,))
        for t in range(T - 2, -1, -1):
            msg = log_trans[t] + log_obs[t + 1][None, :] + beta[None, :]
            beta = jax_logsumexp(msg, axis=1)
            beta = beta - jax_logsumexp(beta)
            betas.append(beta)
        betas = jnp.stack(betas[::-1], axis=0)

        gamma = alphas + betas
        gamma = gamma - jax_logsumexp(gamma, axis=1, keepdims=True)
        gamma = jnp.exp(gamma)

        xis = []
        for t in range(T - 1):
            xi = (
                alphas[t][:, None]
                + log_trans[t]
                + log_obs[t + 1][None, :]
                + betas[t + 1][None, :]
            )
            xi = xi - jax_logsumexp(xi)
            xis.append(jnp.exp(xi))
        if xis:
            xis = jnp.stack(xis, axis=0)
        else:
            xis = jnp.zeros((0, K, K))
        return gamma, xis

    def expected_states_hmm(self, data, global_theta, input=None,
                            mask=None, tag=None, key=None):
        """State posterior using the same time-varying transitions as training."""
        data = jnp.asarray(data)
        inputs = None if input is None else jnp.asarray(input)
        T = data.shape[1]
        log_pi0 = jnp.log(
            jnp.clip(global_theta["init_state_distn"]["pi0"], LOG_EPS, 1.0)
        )

        def one_sequence(xs, seq_inputs):
            log_obs = self.observations.log_likelihoods(
                xs, input=seq_inputs, mask=None, tag=None,
                params=global_theta["observations"],
            )
            log_obs = jnp.nan_to_num(log_obs, neginf=-100.0)
            log_trans = self._input_driven_log_transitions(
                seq_inputs, global_theta["transitions"], T
            )
            gamma, _ = self._forward_backward_pairwise(
                log_pi0, log_trans, log_obs
            )
            return gamma

        if inputs is None:
            return jax.vmap(lambda xs: one_sequence(xs, None))(data)
        return jax.vmap(one_sequence)(data, inputs)


    def _log_likelihood(self, data, inputs, global_theta, local_theta, old_ll,
                        batch_weights=None):
        N, T, D = data.shape
        assert self.D == D, f"`data` must be {self.D}-dimensional..."

        if batch_weights is None:
            weights = jnp.ones((N, T))
        else:
            weights = jnp.asarray(batch_weights)
            if weights.shape != (N, T):
                raise ValueError(
                    f"IPW shape {weights.shape} does not match data {(N, T)}"
                )

        # x_0 is input-independent; x_t uses treatment history through t-1.
        obs_weights = jnp.concatenate(
            [jnp.ones_like(weights[:, :1]), weights[:, :-1]], axis=1
        )

        log_init_dist = jnp.log(jnp.clip(global_theta["init_state_distn"]["pi0"], LOG_EPS, 1.0))

        ll = lambda xs, inp: self.observations.log_likelihoods(
            xs, input=inp, mask=None, tag=None, params=global_theta["observations"]
        )
        if inputs is None:
            log_obs_lik = jnp.stack([ll(data[n], None) for n in range(N)], axis=0)
        else:
            log_obs_lik = jnp.stack([ll(data[n], inputs[n]) for n in range(N)], axis=0)
        log_obs_lik = jnp.nan_to_num(log_obs_lik, neginf=-100.0)

        if self.likelihood_loss == "exact_pairwise":
            total = 0.0
            for n in range(N):
                seq_inputs = None if inputs is None else inputs[n]
                log_trans = self._input_driven_log_transitions(seq_inputs, global_theta["transitions"], T)
                # gamma, xi = self._forward_backward_pairwise(log_init_dist, log_trans, log_obs_lik[n])

                weighted_log_obs = (
                    obs_weights[n, :, None] * log_obs_lik[n]
                )
                weighted_log_trans = (
                    weights[n, :-1, None, None] * log_trans[: T - 1]
                )

                gamma, xi = self._forward_backward_pairwise(
                    log_init_dist,
                    weighted_log_trans,
                    weighted_log_obs,
                )

                init_lik = jnp.sum(gamma[0] * log_init_dist)
                clust_lik = jnp.sum(obs_weights[n, :, None] * gamma * log_obs_lik[n])
                trans_lik = jnp.sum(weights[n, :-1, None, None] * xi * log_trans[: T - 1])
                # total = total + init_lik + clust_lik + trans_lik

                # Safe logs: zero-probability terms contribute 0, without evaluating log(0).
                log_gamma = jnp.log(jnp.where(gamma > 0.0, gamma, 1.0))
                log_xi = jnp.log(jnp.where(xi > 0.0, xi, 1.0))

                # H[q(z_0:T)] = H[q(z_0)] + sum_t H[q(z_{t+1} | z_t)]
                state_entropy = (
                    -jnp.sum(gamma[0] * log_gamma[0])
                    -jnp.sum(
                        xi * (log_xi - log_gamma[:-1, :, None])
                    )
                )

                total = (
                    total
                    + init_lik
                    + clust_lik
                    + trans_lik
                    + state_entropy
                )
            return total

        init_lik = jnp.sum(local_theta[:, 0, :] * log_init_dist[None, :])
        clust_lik = jnp.sum(obs_weights[:, :, None] * log_obs_lik * local_theta)

        trans_lik = 0.0
        for n in range(N):
            seq_inputs = None if inputs is None else inputs[n]
            log_trans = self._input_driven_log_transitions(seq_inputs, global_theta["transitions"], T)
            pair_q = local_theta[n, :-1, :, None] * local_theta[n, 1:, None, :]
            trans_lik = trans_lik + jnp.sum(weights[n, :-1, None, None] * pair_q * log_trans[: T - 1])

        return init_lik + trans_lik + clust_lik


def jax_logsumexp(x, axis=None, keepdims=False):
    m = jnp.max(x, axis=axis, keepdims=True)
    out = m + jnp.log(jnp.sum(jnp.exp(x - m), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        out = jnp.squeeze(out, axis=axis)
    return out


def _normalize_rows_np(P):
    P = np.asarray(P, dtype=float)
    return P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)


def _transition_matrix_from_params(Ps, Ws, input_vec):
    """
    Matches the synthetic generator:
        effects = exp(Ws @ [input, 1])
        P_t(k->k') ∝ Ps(k->k') * effects(k')
    """
    Ps = np.asarray(Ps, dtype=float)
    Ws = np.asarray(Ws, dtype=float)
    aug = np.concatenate([np.asarray(input_vec, dtype=float).reshape(-1), np.ones(1)])

    if aug.shape[0] < Ws.shape[1]:
        aug = np.pad(aug, (0, Ws.shape[1] - aug.shape[0]))
    elif aug.shape[0] > Ws.shape[1]:
        aug = aug[: Ws.shape[1]]

    effects = np.exp(Ws @ aug)
    mod = Ps * effects[None, :]
    return _normalize_rows_np(mod)


def build_propensity_sequence_features(obs, treatment, extra_inputs=None, lags=2):
    """
    Build pre-treatment sequence features for the propensity model.

    Features at time t include:
      - lagged outcomes Y_{t-1}, ..., Y_{t-lags}
      - lagged treatments W_{t-1}, ..., W_{t-lags}
      - optional lagged extra observed channels (e.g. noisy proxy/confounder)

    Returns
    -------
    x_seq : np.ndarray, shape (N, T, F)
    y_seq : np.ndarray, shape (N, T)
        Current treatment labels W_t
    """
    obs = np.asarray(obs, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    N, T, D = obs.shape

    pieces = []
    for lag in range(1, lags + 1):
        past_obs = np.zeros_like(obs)
        if lag < T:
            past_obs[:, lag:, :] = obs[:, :-lag, :]
        pieces.append(past_obs)

        past_treat = np.zeros((N, T, 1), dtype=float)
        if lag < T:
            past_treat[:, lag:, 0] = treatment[:, :-lag]
        pieces.append(past_treat)

        if extra_inputs is not None:
            extra_inputs = np.asarray(extra_inputs, dtype=float)
            E = extra_inputs.shape[-1]
            past_extra = np.zeros((N, T, E), dtype=float)
            if lag < T:
                past_extra[:, lag:, :] = extra_inputs[:, :-lag, :]
            pieces.append(past_extra)

    x_seq = np.concatenate(pieces, axis=-1)
    y_seq = treatment.astype(float)
    return x_seq, y_seq


def fit_history_treatment_model(
    train_obs,
    train_treatment,
    train_extra_inputs=None,
    lags=2,
    model_type="logistic",
    logistic_C=1.0,
    lstm_hidden_dim=32,
    lstm_num_layers=1,
    lstm_lr=1e-3,
    lstm_epochs=100,
):
    """
    Fit propensity model:
      - logistic: flatten (N,T,F) -> (N*T,F)
      - lstm: sequence model over time
    """
    x_seq, y_seq = build_propensity_sequence_features(
        train_obs,
        train_treatment,
        extra_inputs=train_extra_inputs,
        lags=lags,
    )

    if model_type == "logistic":
        X = x_seq.reshape(-1, x_seq.shape[-1])
        y = y_seq.reshape(-1).astype(int)
        model = LogisticRegression(max_iter=1000, C=logistic_C, solver="lbfgs")
        model.fit(X, y)
        return {"type": "logistic", "model": model, "lags": lags}

    if model_type == "lstm":
        x_t = torch.as_tensor(x_seq, dtype=torch.float32)
        y_t = torch.as_tensor(y_seq, dtype=torch.float32)

        model = PropensityLSTM(
            input_dim=x_seq.shape[-1],
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
        )
        opt = torch.optim.Adam(model.parameters(), lr=lstm_lr)

        model.train()
        for _ in range(lstm_epochs):
            opt.zero_grad()
            logits = model(x_t)
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            loss.backward()
            opt.step()

        model.eval()
        return {"type": "lstm", "model": model, "lags": lags}

    raise ValueError(f"Unknown propensity model_type: {model_type}")


def predict_history_treatment_probs(propensity_obj, obs, treatment, extra_inputs=None):
    """
    Return per-time P(W_t=1 | past observed history).
    """
    model_type = propensity_obj["type"]
    lags = propensity_obj["lags"]

    x_seq, _ = build_propensity_sequence_features(
        obs,
        treatment,
        extra_inputs=extra_inputs,
        lags=lags,
    )

    if model_type == "logistic":
        model = propensity_obj["model"]
        X = x_seq.reshape(-1, x_seq.shape[-1])
        p = model.predict_proba(X)[:, 1].reshape(x_seq.shape[0], x_seq.shape[1])

    elif model_type == "lstm":
        model = propensity_obj["model"]
        x_t = torch.as_tensor(x_seq, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x_t)
            p = torch.sigmoid(logits).cpu().numpy()

    else:
        raise ValueError(f"Unknown propensity model type: {model_type}")

    return np.clip(p, 1e-2, 1.0 - 1e-2)


def sequential_inverse_probability_weights(treatment, probs, clip=50.0):
    treatment = np.asarray(treatment, dtype=float)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-2, 1.0 - 1e-2)
    contrib = np.where(treatment > 0.5, 1.0 / probs, 1.0 / (1.0 - probs))
    alpha = np.cumprod(contrib, axis=1)
    if clip is not None:
        alpha = np.clip(alpha, 0.05, clip) # todo take both min and max
    return alpha


def summarize_weights(weights):
    ess = []
    for t in range(weights.shape[1]):
        w = weights[:, t]
        sw = np.sum(w)
        ess_t = (sw**2) / max(np.sum(w**2), 1e-12)
        ess.append(ess_t)
    return {
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_ess_mean": float(np.mean(ess)),
    }


def fit_model(
    obs_split,
    tags_split,
    treatment_split,
    hmm_inputs_split,
    result_dir,
    M,
    seed=1234,
    num_iters=100,
    use_history_weights=False,
    history_lags=2,
    weight_clip=50.0,
    likelihood_loss="exact_pairwise",
    propensity_model_type="logistic",
    propensity_extra_split=None,
    propensity_lstm_hidden_dim=32,
    propensity_lstm_num_layers=1,
    propensity_lstm_lr=1e-3,
    propensity_lstm_epochs=100,
    num_states=2,
    batch_size=None,
    output_preprocessing=None,
    covariance_prior_jitter=0.0,
    early_stopping=False,
    early_stopping_patience=5,
    early_stopping_min_delta=0.0,
    validation_batch_size=64,
    validation_seed=20260929,
):
    if not np.isfinite(covariance_prior_jitter) or covariance_prior_jitter < 0:
        raise ValueError("covariance_prior_jitter must be finite and nonnegative")
    # Infer dimensions from the supplied data so this entry point also accepts
    # processed MIMIC splits, rather than assuming the synthetic D=2 dataset.
    for name, splits in (
        ("obs_split", obs_split),
        ("tags_split", tags_split),
        ("treatment_split", treatment_split),
        ("hmm_inputs_split", hmm_inputs_split),
    ):
        if len(splits) != 3:
            raise ValueError(f"{name} must contain train, validation, and test splits")
    if not isinstance(M, (int, np.integer)) or M < 0:
        raise ValueError("M must be a nonnegative integer")
    if not isinstance(num_states, (int, np.integer)) or num_states < 1:
        raise ValueError("num_states must be a positive integer")

    obs_split = tuple(np.asarray(x) for x in obs_split)
    tags_split = tuple(np.asarray(x) for x in tags_split)
    treatment_split = tuple(np.asarray(x) for x in treatment_split)
    hmm_inputs_split = tuple(
        None if x is None else np.asarray(x) for x in hmm_inputs_split
    )
    data_shape = None
    for name, obs, tags, treatment, inputs in zip(
        ("train", "validation", "test"),
        obs_split, tags_split, treatment_split, hmm_inputs_split,
    ):
        if obs.ndim != 3 or min(obs.shape) < 2:
            raise ValueError(f"{name} observations must have shape (N, T, D) with N, T, D >= 2")
        if data_shape is None:
            data_shape = obs.shape[1:]
        if obs.shape[1:] != data_shape or not np.all(np.isfinite(obs)):
            raise ValueError(f"{name} observations must be finite and share the same T and D")
        if tags.shape != (obs.shape[0],):
            raise ValueError(f"{name} labels must have shape ({obs.shape[0]},)")
        if treatment.shape != obs.shape[:2] or not np.all(np.isfinite(treatment)):
            raise ValueError(f"{name} treatment must be finite with shape {obs.shape[:2]}")
        if use_history_weights and not np.all(np.isin(treatment, (0, 1))):
            raise ValueError(
                f"{name} propensity treatment must be binary 0/1; "
                "do not pass a standardized or continuous bolus dose"
            )
        if inputs is None:
            if M != 0:
                raise ValueError(f"{name} inputs are required when M > 0")
        elif inputs.shape != (*obs.shape[:2], M) or not np.all(np.isfinite(inputs)):
            raise ValueError(f"{name} inputs must be finite with shape {(*obs.shape[:2], M)}")

    split_sizes = [len(x) for x in obs_split]
    if batch_size is None:
        batch_size = min(64, min(split_sizes))
    if not isinstance(batch_size, (int, np.integer)) or not 1 <= batch_size <= min(split_sizes):
        raise ValueError("batch_size must be a positive integer no larger than any split")

    if propensity_extra_split is not None:
        if len(propensity_extra_split) != 3:
            raise ValueError("propensity_extra_split must contain three splits")
        propensity_extra_split = tuple(np.asarray(x) for x in propensity_extra_split)
        extra_dim = None
        for obs, extra in zip(obs_split, propensity_extra_split):
            if extra.ndim != 3 or extra.shape[:2] != obs.shape[:2] or not np.all(np.isfinite(extra)):
                raise ValueError("Propensity extra inputs must be finite with shape (N, T, E)")
            if extra_dim is None:
                extra_dim = extra.shape[-1]
            if extra.shape[-1] != extra_dim:
                raise ValueError("Propensity extra inputs must share the same E across splits")

    C = make_config(result_dir, M=M, seed=seed, num_iters=num_iters, batch_size=batch_size)
    C.K_train = num_states
    C.T, C.D = data_shape
    C.D_lstm = C.D
    C.N_train, C.N_val, C.N_test = split_sizes
    C.N = sum(split_sizes)
    C.use_recognition_marginals = (likelihood_loss == "factorized")
    C.covariance_prior_jitter = float(covariance_prior_jitter)
    C.early_stopping = bool(early_stopping)
    C.early_stopping_patience = early_stopping_patience
    C.early_stopping_min_delta = early_stopping_min_delta
    C.validation_batch_size = validation_batch_size
    C.validation_seed = validation_seed
    hyperparams_dict = {k: v[0] for k, v in C.hyperparameters.items()}

    train_obs, val_obs, test_obs = obs_split
    train_tags, val_tags, test_tags = tags_split

    train_treat, val_treat, test_treat = treatment_split
    train_inputs, val_inputs, test_inputs = hmm_inputs_split

    if propensity_extra_split is None:
        train_extra = val_extra = test_extra = None
    else:
        train_extra, val_extra, test_extra = propensity_extra_split

    os.makedirs(result_dir, exist_ok=True)

    weight_summary = None

    train_w = val_w = test_w = None

    if use_history_weights:
        trt_model = fit_history_treatment_model(
            train_obs=train_obs,
            train_treatment=train_treat,
            train_extra_inputs=train_extra,
            lags=history_lags,
            model_type=propensity_model_type,
            lstm_hidden_dim=propensity_lstm_hidden_dim,
            lstm_num_layers=propensity_lstm_num_layers,
            lstm_lr=propensity_lstm_lr,
            lstm_epochs=propensity_lstm_epochs,
        )

        train_probs = predict_history_treatment_probs(trt_model, train_obs, train_treat, extra_inputs=train_extra)
        val_probs = predict_history_treatment_probs(trt_model, val_obs, val_treat, extra_inputs=val_extra)
        test_probs = predict_history_treatment_probs(trt_model, test_obs, test_treat, extra_inputs=test_extra)

        train_w = sequential_inverse_probability_weights(train_treat, train_probs, clip=weight_clip)
        val_w = sequential_inverse_probability_weights(val_treat, val_probs, clip=weight_clip)
        test_w = sequential_inverse_probability_weights(test_treat, test_probs, clip=weight_clip)

        #todo : try with potential normalization of w: take weights and divide by sum to have L1 norm 1

        weight_summary = summarize_weights(train_w)

        with open(os.path.join(result_dir, "weight_summary.json"), "w") as f:
            json.dump(weight_summary, f, indent=2)

        with open(os.path.join(result_dir, "treatment_model.pkl"), "wb") as f:
            pickle.dump(trt_model, f)

        np.savez(
            os.path.join(result_dir, "treatment_probs.npz"),
            train_probs=train_probs,
            val_probs=val_probs,
            test_probs=test_probs,
        )

    key = random.PRNGKey(seed)
    advi_hmm = WeightedCausalADVIHMM(
        C.K_train,
        C.D,
        C.D_lstm,
        S=C.S,
        C=C.C,
        M=M,
        constrained=False,
        seed=seed,
        observations=C.dist,
        hyperparams=({**hyperparams_dict}, None),
        observation_kwargs={
            "lags": C.lags,
            "prior_scale_sigmasq": 1.0,
            "prior_df_sigmasq": 3.0,
            "prior_lkj_concentration": 1.0,
            "covariance_prior_jitter": C.covariance_prior_jitter,
        },
        transition_kwargs={"dirichlet_concentration": hyperparams_dict["dirichlet_concentration"]},
        likelihood_loss=likelihood_loss,
    )

    save_file = os.path.join(result_dir, "params.pkl")
    cluster_file = os.path.join(result_dir, "cluster_assignments.pkl")

    # Keep each stay's IPW alongside its corresponding data split.
    advi_hmm.causal_ipw_weights = (train_w, val_w, test_w)

    losses, *_ = advi_hmm.fit(
        datas=train_obs,
        labels=train_tags,
        static=None,
        inputs=train_inputs,
        masks=None,
        val_datas=val_obs,
        val_labels=val_tags,
        val_static=None,
        val_inputs=val_inputs,
        val_masks=None,
        test_datas=test_obs,
        test_labels=test_tags,
        test_static=None,
        test_inputs=test_inputs,
        test_masks=None,
        classif_train_datas=train_obs,
        classif_train_labels=train_tags,
        classif_train_static=None,
        classif_train_inputs=train_inputs,
        classif_val_datas=val_obs,
        classif_val_labels=val_tags,
        classif_val_static=None,
        classif_val_inputs=val_inputs,
        key=key,
        verbose=1,
        num_iters=C.num_iters,
        method="advi",
        discriminator_constraint=False,
        similarity_constraint=False,
        true_hmm=None,
        init_true_hmm=False,
        save_file=save_file,
        cluster_assignment_file=cluster_file,
        trained_lstm_file="",
        trained_lstm_config="",
        batch_size=C.BATCH_SIZE,
        sample_global=True,
        sample_local=True,
        use_recognition_marginals=C.use_recognition_marginals,
        old_ll=False,
        early_stopping=C.early_stopping,
        early_stopping_patience=C.early_stopping_patience,
        early_stopping_min_delta=C.early_stopping_min_delta,
        validation_batch_size=C.validation_batch_size,
        validation_seed=C.validation_seed,
    )

    with open(os.path.join(result_dir, "losses.pkl"), "wb") as f:
        pickle.dump(losses, f)

    # Optional post-training interpretation. Training and inference
    # remain on the preprocessed model scale.
    if output_preprocessing is not None:
        from advi_ssm_mimic4.result_scripts.simulate_covariates import (
            plot_covariates,
        )

        observation_features = list(output_preprocessing["features"])
        scaler, pt = output_preprocessing["transforms"]
        preprocessing_info = output_preprocessing["information"]

        if len(observation_features) != C.D:
            raise ValueError(
                "Observation feature names must match the model's D."
            )

        # Preserve the exact training transforms beside the fitted model.
        with open(
            os.path.join(result_dir, "output_preprocessing.pkl"), "wb"
        ) as f:
            pickle.dump(output_preprocessing, f)

        global_theta = _generate_global_theta(
            save_file, advi_hmm, cluster_file
        )

        plot_covariates(
            advi_hmm,
            global_theta["observations"],
            C.D,
            num_sample_timesteps=C.T,
            save=True,
            save_dir=os.path.join(result_dir, "observation_interpretation"),
            num_avg_repeats=20,
            features_list=observation_features,
            scaler=scaler,
            pt=pt,
            preprocessing_info=preprocessing_info,
        )
    return advi_hmm, save_file, C, weight_summary, cluster_file


def state_accuracy(advi_hmm, save_file, cluster_file, C, obs_split, inputs_split, truth):
    test_obs = np.asarray(obs_split[2])
    test_inputs = np.asarray(inputs_split[2]) if inputs_split is not None else None

    global_theta = _generate_global_theta(save_file, advi_hmm, cluster_file)
    local_theta = advi_hmm.expected_states_hmm(
        test_obs, global_theta, input=test_inputs
    )

    pred_states = np.asarray(np.argmax(local_theta, axis=-1))
    true_states = np.asarray(truth["states_true_test"])
    perm = _find_permutation(
        true_states.reshape(-1).astype(int),
        pred_states.reshape(-1).astype(int),
        K1=C.K_true,
        K2=C.K_train,
    )
    perm_map = {int(perm[i]): int(i) for i in range(len(perm))}
    pred_aligned = np.vectorize(lambda z: perm_map.get(int(z), int(z)))(pred_states)
    return {
        "state_acc_raw": float((pred_states == true_states).mean()),
        "state_acc_aligned": float((pred_aligned == true_states).mean()),
    }


def _pred_to_true_perm_map(perm):
    return {int(pred): int(true) for true, pred in enumerate(np.asarray(perm).tolist())}


def _align_state_labels(pred_states, perm):
    pred_to_true = _pred_to_true_perm_map(perm)
    return np.vectorize(lambda z: pred_to_true.get(int(z), int(z)))(np.asarray(pred_states))


def _reorder_state_first_axis(arr, perm):
    arr = np.asarray(arr)
    perm = np.asarray(perm, dtype=int)
    if arr.ndim == 0 or arr.shape[0] < len(perm):
        return arr
    return arr[perm]


def _reorder_transition_matrix(mat, perm):
    mat = np.asarray(mat)
    perm = np.asarray(perm, dtype=int)
    if mat.ndim != 2 or min(mat.shape) < len(perm):
        return mat
    return mat[np.ix_(perm, perm)]


def _reorder_transition_tensor(delta, perm):
    delta = np.asarray(delta)
    perm = np.asarray(perm, dtype=int)
    if delta.ndim < 3 or delta.shape[-1] < len(perm) or delta.shape[-2] < len(perm):
        return delta
    return delta[..., perm, :][..., :, perm]


def _safe_corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return None
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _mse_mae_metrics(name, fitted, true):
    fitted = np.asarray(fitted)
    true = np.asarray(true)
    if fitted.shape != true.shape:
        return {}
    return {
        f"{name}_mse": float(np.mean((fitted - true) ** 2)),
        f"{name}_mae": float(np.mean(np.abs(fitted - true))),
    }


def _treatment_specific_transition_matrices(Ps, Ws, conf_vals=None, observed_mode="full_inputs"):
    """
    Returns treatment-specific transition matrices P(W=0), P(W=1).

    For full_inputs:
        uses per-sample confounder/proxy values
    For treatment_only:
        fitted model has no confounder input, so returns constant matrices
        broadcast across conf_vals if provided
    """
    Ps = np.asarray(Ps)
    Ws = np.asarray(Ws)

    if observed_mode == "treatment_only":
        P0 = _transition_matrix_from_params(Ps, Ws, [0.0])
        P1 = _transition_matrix_from_params(Ps, Ws, [1.0])
        if conf_vals is None:
            return P0[None, ...], P1[None, ...]
        n = np.asarray(conf_vals).reshape(-1).shape[0]
        P0 = np.broadcast_to(P0, (n,) + P0.shape).copy()
        P1 = np.broadcast_to(P1, (n,) + P1.shape).copy()
        return P0, P1

    conf_vals = np.asarray(conf_vals).reshape(-1)
    P0 = np.stack([_transition_matrix_from_params(Ps, Ws, [0.0, u]) for u in conf_vals], axis=0)
    P1 = np.stack([_transition_matrix_from_params(Ps, Ws, [1.0, u]) for u in conf_vals], axis=0)
    return P0, P1


def evaluate_recovery(advi_hmm, save_file, cluster_file, C, obs_split, inputs_split, truth, observed_mode="full_inputs"):
    metrics = state_accuracy(advi_hmm, save_file, cluster_file, C, obs_split, inputs_split, truth)

    test_obs = np.asarray(obs_split[2])
    test_inputs = np.asarray(inputs_split[2]) if inputs_split is not None else None
    true_states = np.asarray(truth["states_true_test"])

    global_theta = _generate_global_theta(save_file, advi_hmm, cluster_file)
    local_theta = advi_hmm.expected_states_hmm(
        test_obs, global_theta, input=test_inputs
    )

    pred_states = np.asarray(np.argmax(local_theta, axis=-1))

    perm = _find_permutation(
        true_states.reshape(-1).astype(int),
        pred_states.reshape(-1).astype(int),
        K1=C.K_true,
        K2=C.K_train,
    )
    metrics["state_perm"] = [int(x) for x in np.asarray(perm).tolist()]

    if "transitions" in global_theta:
        trans = global_theta["transitions"]

        if "Ps" in trans and "Ps_true" in truth:
            fitted_Ps = _reorder_transition_matrix(np.asarray(trans["Ps"]), perm)
            metrics.update(_mse_mae_metrics("Ps", fitted_Ps, np.asarray(truth["Ps_true"])))

        if "Ws" in trans and "Ws_true" in truth:
            fitted_Ws = _reorder_state_first_axis(np.asarray(trans["Ws"]), perm)
            if fitted_Ws.shape == np.asarray(truth["Ws_true"]).shape:
                metrics.update(_mse_mae_metrics("Ws", fitted_Ws, np.asarray(truth["Ws_true"])))

        if "Ps" in trans and "Ws" in trans and "Ps_true" in truth and "Ws_true" in truth and "confounder_true_test" in truth:
            Ps_hat_raw = np.asarray(trans["Ps"])
            Ws_hat_raw = np.asarray(trans["Ws"])

            Ps_true = np.asarray(truth["Ps_true"])
            Ws_true = np.asarray(truth["Ws_true"])

            conf_true_vals = np.asarray(truth["confounder_true_test"]).reshape(-1)

            P0_true, P1_true = _treatment_specific_transition_matrices(
                Ps_true, Ws_true, conf_vals=conf_true_vals, observed_mode="full_inputs"
            )

            if observed_mode == "treatment_only":
                P0_hat, P1_hat = _treatment_specific_transition_matrices(
                    Ps_hat_raw, Ws_hat_raw, conf_vals=conf_true_vals, observed_mode="treatment_only"
                )
            else:
                if test_inputs is None or test_inputs.shape[-1] < 2:
                    P0_hat, P1_hat = _treatment_specific_transition_matrices(
                        Ps_hat_raw, Ws_hat_raw, conf_vals=conf_true_vals, observed_mode="treatment_only"
                    )
                else:
                    conf_fit_vals = np.asarray(test_inputs)[..., 1].reshape(-1)
                    P0_hat, P1_hat = _treatment_specific_transition_matrices(
                        Ps_hat_raw, Ws_hat_raw, conf_vals=conf_fit_vals, observed_mode="full_inputs"
                    )

            P0_hat = _reorder_transition_tensor(P0_hat, perm)
            P1_hat = _reorder_transition_tensor(P1_hat, perm)

            delta_true = P1_true - P0_true
            delta_hat = P1_hat - P0_hat

            metrics.update(_mse_mae_metrics("delta_transition", delta_hat, delta_true))
            

            P0_true_avg = np.mean(P0_true, axis=0)
            P1_true_avg = np.mean(P1_true, axis=0)
            P0_hat_avg = np.mean(P0_hat, axis=0)
            P1_hat_avg = np.mean(P1_hat, axis=0)

            metrics.update(_mse_mae_metrics("P0", P0_hat_avg, P0_true_avg))
            metrics.update(_mse_mae_metrics("P1", P1_hat_avg, P1_true_avg))

            metrics["P0_true_avg"] = P0_true_avg.tolist()
            metrics["P1_true_avg"] = P1_true_avg.tolist()
            metrics["P0_hat_avg"] = P0_hat_avg.tolist()
            metrics["P1_hat_avg"] = P1_hat_avg.tolist()
            

    if "observations" in global_theta:
        obs = global_theta["observations"]
        if "As" in obs and "As_true" in truth:
            fitted_As = _reorder_state_first_axis(np.asarray(obs["As"]), perm)
            metrics.update(_mse_mae_metrics("As", fitted_As, np.asarray(truth["As_true"])))
        if "bs" in obs and "bs_true" in truth:
            fitted_bs = _reorder_state_first_axis(np.asarray(obs["bs"]), perm)
            metrics.update(_mse_mae_metrics("bs", fitted_bs, np.asarray(truth["bs_true"])))
        if "Vs" in obs and "Vs_true" in truth:
            fitted_Vs = _reorder_state_first_axis(np.asarray(obs["Vs"]), perm)
            if fitted_Vs.shape == np.asarray(truth["Vs_true"]).shape:
                metrics.update(_mse_mae_metrics("Vs", fitted_Vs, np.asarray(truth["Vs_true"])))

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="causal_iohmm_benchmark")
    parser.add_argument("--num-iters", type=int, default=75)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--history-lags", type=int, default=2)
    parser.add_argument("--weight-clip", type=float, default=50.0)
    parser.add_argument("--use-history-weights", action="store_true")
    parser.add_argument("--likelihood-loss", choices=["factorized", "exact_pairwise"], default="exact_pairwise")

    parser.add_argument("--propensity-model", choices=["logistic", "lstm"], default="logistic")
    parser.add_argument("--propensity-lstm-hidden-dim", type=int, default=32)
    parser.add_argument("--propensity-lstm-num-layers", type=int, default=1)
    parser.add_argument("--propensity-lstm-lr", type=float, default=1e-3)
    parser.add_argument("--propensity-lstm-epochs", type=int, default=100)

    args = parser.parse_args()

    result_root = args.result_root
    os.makedirs(result_root, exist_ok=True)

    base_cfg = SimpleNamespace(N=256, T=24, K_true=2, D=2, N_train=128, N_val=64, seed=args.seed)
    obs, tags, oracle_inputs, truth = simulate_causal_iohmm(base_cfg, observed_confounder="oracle")
    _, _, proxy_inputs, truth_proxy = simulate_causal_iohmm(base_cfg, observed_confounder="proxy")

    truth_proxy.update(
        {
            k: v
            for k, v in truth.items()
            if k.endswith("_true")
            or k
            in (
                "params",
                "z_true",
                "states_true",
                "confounder_true",
                "transition_effect_true",
                "emission_effect_true",
            )
        }
    )

    obs_split, tags_split, _, oracle_split, _, split_truth = split_synthetic(obs, tags, oracle_inputs, truth, base_cfg)
    _, _, _, proxy_split, _, split_truth_proxy = split_synthetic(obs, tags, proxy_inputs, truth_proxy, base_cfg)

    # HMM inputs
    treat_only_split = tuple(x[..., :1] for x in proxy_split)

    # Propensity inputs
    treatment_split = tuple(x[..., 0] for x in proxy_split)  # (N,T)
    propensity_extra_split = tuple(x[..., 1:] for x in proxy_split)  # proxy/current confounder channel

    save_truth_npz(os.path.join(result_root, "oracle_truth.npz"), split_truth)
    save_truth_npz(os.path.join(result_root, "proxy_truth.npz"), split_truth_proxy)

    variants = [
        ("treatment_only", treat_only_split, 1, False, "treatment_only"),
    ]
    if args.use_history_weights:
        variants.extend(
            [
                ("treatment_only_history_weighted", treat_only_split, 1, True, "treatment_only"),
                ("full_inputs_history_weighted", proxy_split, 2, True, "full_inputs"),
            ]
        )

    metrics = {}
    for name, hmm_inputs_split, M, weighted, observed_mode in variants:
        run_dir = os.path.join(result_root, name)

        advi_hmm, save_file, C, weight_summary, cluster_file = fit_model(
            obs_split=obs_split,
            tags_split=tags_split,
            treatment_split=treatment_split,
            hmm_inputs_split=hmm_inputs_split,
            result_dir=run_dir,
            M=M,
            seed=args.seed,
            num_iters=args.num_iters,
            use_history_weights=weighted,
            history_lags=args.history_lags,
            weight_clip=args.weight_clip,
            likelihood_loss=args.likelihood_loss,
            propensity_model_type=args.propensity_model,
            propensity_extra_split=propensity_extra_split,
            propensity_lstm_hidden_dim=args.propensity_lstm_hidden_dim,
            propensity_lstm_num_layers=args.propensity_lstm_num_layers,
            propensity_lstm_lr=args.propensity_lstm_lr,
            propensity_lstm_epochs=args.propensity_lstm_epochs,
        )

        run_metrics = evaluate_recovery(
            advi_hmm=advi_hmm,
            save_file=save_file,
            cluster_file=cluster_file,
            C=C,
            obs_split=obs_split,
            inputs_split=hmm_inputs_split,
            truth=(split_truth if observed_mode == "treatment_only" else split_truth_proxy),
            observed_mode=observed_mode,
        )

        run_metrics["likelihood_loss"] = args.likelihood_loss
        if weighted:
            run_metrics["propensity_model"] = args.propensity_model
        if weight_summary is not None:
            run_metrics.update(weight_summary)

        metrics[name] = run_metrics
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(metrics[name], f, indent=2)

    with open(os.path.join(result_root, "comparison.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
