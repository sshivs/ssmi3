import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

import causal_iohmm as sc


def make_external_config_factory(obs, inputs, states, labels, K_true, D, M, T):
    def make_external_config(result_dir, M, seed=1234, num_iters=30, batch_size=128):
        return SimpleNamespace(
            K_true=K_true,
            K_train=K_true,
            D=D,
            D_lstm=D,
            S=0,
            C=2,
            M=M,
            T=T,
            N=obs.shape[0],
            N_train=0,
            N_val=0,
            N_test=0,
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

    return make_external_config

from sklearn.model_selection import train_test_split

def split_external_data(obs, inputs, labels, states, n_train=128, n_val=64, seed=1234):
    labels = np.asarray(labels)
    idx = np.arange(obs.shape[0])

    # train split
    train_idx, rest_idx = train_test_split(
        idx,
        train_size=n_train,
        random_state=seed,
        stratify=labels,
    )

    # val/test split
    rest_labels = labels[rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=n_val,
        random_state=seed,
        stratify=rest_labels,
    )

    obs_split = (obs[train_idx], obs[val_idx], obs[test_idx])
    tags_split = (labels[train_idx], labels[val_idx], labels[test_idx])
    inputs_split = (inputs[train_idx], inputs[val_idx], inputs[test_idx])

    truth = {
        "states_true_test": states[test_idx],
    }
    return obs_split, tags_split, inputs_split, truth




def reorder_state_first_axis(arr, perm):
    arr = np.asarray(arr)
    perm = np.asarray(perm, dtype=int)
    if arr.ndim == 0 or arr.shape[0] < len(perm):
        return arr
    return arr[perm]


def reorder_transition_matrix(mat, perm):
    mat = np.asarray(mat)
    perm = np.asarray(perm, dtype=int)
    if mat.ndim != 2:
        return mat
    return mat[np.ix_(perm, perm)]


def reorder_transition_tensor(delta, perm):
    delta = np.asarray(delta)
    perm = np.asarray(perm, dtype=int)
    if delta.ndim < 3:
        return delta
    return delta[..., perm, :][..., :, perm]


def mse_mae(name, fitted, true):
    fitted = np.asarray(fitted)
    true = np.asarray(true)
    if fitted.shape != true.shape:
        return {}
    return {
        f"{name}_mse": float(np.mean((fitted - true) ** 2)),
        f"{name}_mae": float(np.mean(np.abs(fitted - true))),
    }


def safe_corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.shape != b.shape or a.size == 0 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def treatment_specific_mats_from_two_inputs(Ps, Ws, other_input_vals):
    """
    Treat input channel 0 as treatment of interest.
    Hold input channel 1 fixed at observed values.
    """
    P0 = np.stack(
        [sc._transition_matrix_from_params(Ps, Ws, [0.0, float(v)]) for v in other_input_vals],
        axis=0,
    )
    P1 = np.stack(
        [sc._transition_matrix_from_params(Ps, Ws, [1.0, float(v)]) for v in other_input_vals],
        axis=0,
    )
    return P0, P1


def dry_run_shape_check(obs_split, treatment_split, hmm_inputs_split, full_inputs_split, bundle, m_dim, observed_mode):
    train_obs, val_obs, test_obs = [np.asarray(x) for x in obs_split]
    train_treat, val_treat, test_treat = [np.asarray(x) for x in treatment_split]
    train_hmm, val_hmm, test_hmm = [np.asarray(x) for x in hmm_inputs_split]
    train_full, val_full, test_full = [np.asarray(x) for x in full_inputs_split]

    # basic rank checks
    assert train_obs.ndim == 3 and val_obs.ndim == 3 and test_obs.ndim == 3
    assert train_treat.ndim == 2 and val_treat.ndim == 2 and test_treat.ndim == 2
    assert train_hmm.ndim == 3 and val_hmm.ndim == 3 and test_hmm.ndim == 3
    assert train_full.ndim == 3 and val_full.ndim == 3 and test_full.ndim == 3

    for obs_arr, trt_arr, hmm_arr, full_arr, name in [
        (train_obs, train_treat, train_hmm, train_full, "train"),
        (val_obs, val_treat, val_hmm, val_full, "val"),
        (test_obs, test_treat, test_hmm, test_full, "test"),
    ]:
        assert obs_arr.shape[:2] == trt_arr.shape[:2], f"{name}: obs vs treatment mismatch {obs_arr.shape} vs {trt_arr.shape}"
        assert obs_arr.shape[:2] == hmm_arr.shape[:2], f"{name}: obs vs hmm_inputs mismatch {obs_arr.shape} vs {hmm_arr.shape}"
        assert obs_arr.shape[:2] == full_arr.shape[:2], f"{name}: obs vs full_inputs mismatch {obs_arr.shape} vs {full_arr.shape}"

    D = int(bundle["D"])
    assert train_obs.shape[-1] == D, f"Expected obs dim {D}, got {train_obs.shape[-1]}"
    assert val_obs.shape[-1] == D, f"Expected obs dim {D}, got {val_obs.shape[-1]}"
    assert test_obs.shape[-1] == D, f"Expected obs dim {D}, got {test_obs.shape[-1]}"

    assert train_hmm.shape[-1] == m_dim, f"Expected HMM input dim {m_dim}, got {train_hmm.shape[-1]}"
    assert val_hmm.shape[-1] == m_dim, f"Expected HMM input dim {m_dim}, got {val_hmm.shape[-1]}"
    assert test_hmm.shape[-1] == m_dim, f"Expected HMM input dim {m_dim}, got {test_hmm.shape[-1]}"

    assert train_full.shape[-1] >= 2, "Need at least 2 full input channels for notebook evaluation"
    assert val_full.shape[-1] >= 2, "Need at least 2 full input channels for notebook evaluation"
    assert test_full.shape[-1] >= 2, "Need at least 2 full input channels for notebook evaluation"

    assert np.allclose(train_treat, train_full[..., 0]), "train treatment_split != full_inputs[...,0]"
    assert np.allclose(val_treat, val_full[..., 0]), "val treatment_split != full_inputs[...,0]"
    assert np.allclose(test_treat, test_full[..., 0]), "test treatment_split != full_inputs[...,0]"

    if observed_mode == "treatment_only":
        assert train_hmm.shape[-1] == 1, "treatment_only branch must use 1-channel HMM input"
    elif observed_mode == "full_inputs":
        assert train_hmm.shape[-1] == train_full.shape[-1], "full_inputs branch must use all notebook inputs"
    else:
        raise ValueError(f"Unknown observed_mode: {observed_mode}")

    print(
        f"[dry-run ok] mode={observed_mode}, "
        f"obs_dim={D}, hmm_input_dim={m_dim}, full_input_dim={train_full.shape[-1]}"
    )


def evaluate_external_bundle(
    standalone_module,
    advi_hmm,
    save_file,
    cluster_file,
    C,
    obs_split,
    hmm_inputs_split,
    full_inputs_split,
    truth,
    bundle,
    observed_mode="full_inputs",
):
    test_obs = np.asarray(obs_split[2])
    test_inputs_hmm = np.asarray(hmm_inputs_split[2]) if hmm_inputs_split is not None else None
    test_inputs_full = np.asarray(full_inputs_split[2])
    true_states = np.asarray(truth["states_true_test"])

    global_theta = standalone_module._generate_global_theta(save_file, advi_hmm, cluster_file)

    # Evaluate the same time-varying HMM posterior used by exact_pairwise.
    # Keep the input dimensionality identical to the fitted HMM.
    local_theta = advi_hmm.expected_states_hmm(
        test_obs, global_theta, input=test_inputs_hmm
    )
    pred_states = np.asarray(np.argmax(local_theta, axis=-1))

    perm = standalone_module._find_permutation(
        true_states.reshape(-1).astype(int),
        pred_states.reshape(-1).astype(int),
        K1=C.K_true,
        K2=C.K_train,
    )

    perm_map = {int(perm[i]): int(i) for i in range(len(perm))}
    pred_aligned = np.vectorize(lambda z: perm_map.get(int(z), int(z)))(pred_states)

    metrics = {
        "state_acc_raw": float((pred_states == true_states).mean()),
        "state_acc_aligned": float((pred_aligned == true_states).mean()),
        "state_perm": [int(x) for x in np.asarray(perm).tolist()],
    }


    # truth params from exported notebook bundle
    Ps_true = np.asarray(bundle["Ps_true"])
    Ws_true = np.asarray(bundle["Ws_true"])
    As_true = np.asarray(bundle["As_true"])
    bs_true = np.asarray(bundle["bs_true"])

    trans = global_theta["transitions"]
    obs = global_theta["observations"]

    metrics['Ps_trans'] = np.asarray(trans["Ps"]).tolist()

    Ps_hat = reorder_transition_matrix(np.asarray(trans["Ps"]), perm)
    metrics.update(mse_mae("Ps", Ps_hat, Ps_true))

    Ws_hat_raw = np.asarray(trans["Ws"])
    if Ws_hat_raw.shape == Ws_true.shape:
        Ws_hat = reorder_state_first_axis(Ws_hat_raw, perm)
        metrics.update(mse_mae("Ws", Ws_hat, Ws_true))

    metrics['Ws_trans'] = np.asarray(trans["Ws"]).tolist()

    As_hat = reorder_state_first_axis(np.asarray(obs["As"]), perm)
    bs_hat = reorder_state_first_axis(np.asarray(obs["bs"]), perm)
    metrics.update(mse_mae("As", As_hat, As_true))
    metrics.update(mse_mae("bs", bs_hat, bs_true))

    if "Vs_true" in bundle.files and "Vs" in obs:
        Vs_true = np.asarray(bundle["Vs_true"])
        Vs_hat = reorder_state_first_axis(np.asarray(obs["Vs"]), perm)
        if Vs_hat.shape == Vs_true.shape:
            metrics.update(mse_mae("Vs", Vs_hat, Vs_true))

    if "mu_init_true" in bundle.files and "mu_init" in obs:
        mu_true = np.asarray(bundle["mu_init_true"])
        mu_hat = reorder_state_first_axis(np.asarray(obs["mu_init"]), perm)
        if mu_hat.shape == mu_true.shape:
            metrics.update(mse_mae("mu_init", mu_hat, mu_true))

    # treatment-specific transition matrices:
    # channel 0 = treatment of interest
    # channel 1 = second observed input held fixed
    other_vals_true = test_inputs_full[..., 1].reshape(-1)
    P0_true, P1_true = treatment_specific_mats_from_two_inputs(Ps_true, Ws_true, other_vals_true)

    Ps_hat_raw = np.asarray(trans["Ps"])
    Ws_hat_raw = np.asarray(trans["Ws"])

    if observed_mode == "full_inputs":
        other_vals_hat = test_inputs_hmm[..., 1].reshape(-1)
        P0_hat, P1_hat = treatment_specific_mats_from_two_inputs(Ps_hat_raw, Ws_hat_raw, other_vals_hat)
    elif observed_mode == "treatment_only":
        P0_single = standalone_module._transition_matrix_from_params(Ps_hat_raw, Ws_hat_raw, [0.0])
        P1_single = standalone_module._transition_matrix_from_params(Ps_hat_raw, Ws_hat_raw, [1.0])
        n_eval = other_vals_true.shape[0]
        P0_hat = np.broadcast_to(P0_single, (n_eval,) + P0_single.shape).copy()
        P1_hat = np.broadcast_to(P1_single, (n_eval,) + P1_single.shape).copy()
    else:
        raise ValueError(f"Unknown observed_mode: {observed_mode}")

    P0_hat = reorder_transition_tensor(P0_hat, perm)
    P1_hat = reorder_transition_tensor(P1_hat, perm)

    delta_true = P1_true - P0_true
    delta_hat = P1_hat - P0_hat

    metrics.update(mse_mae("delta_transition", delta_hat, delta_true))
    corr = safe_corr(delta_hat, delta_true)
    if corr is not None:
        metrics["delta_transition_corr"] = corr

    P0_true_avg = np.mean(P0_true, axis=0)
    P1_true_avg = np.mean(P1_true, axis=0)
    P0_hat_avg = np.mean(P0_hat, axis=0)
    P1_hat_avg = np.mean(P1_hat, axis=0)

    metrics.update(mse_mae("P0", P0_hat_avg, P0_true_avg))
    metrics.update(mse_mae("P1", P1_hat_avg, P1_true_avg))
    metrics["P0_true_avg"] = P0_true_avg.tolist()
    metrics["P1_true_avg"] = P1_true_avg.tolist()
    metrics["P0_hat_avg"] = P0_hat_avg.tolist()
    metrics["P1_hat_avg"] = P1_hat_avg.tolist()
    metrics["delta_transition_avg_true"] = (P1_true_avg - P0_true_avg).tolist()
    metrics["delta_transition_avg_hat"] = (P1_hat_avg - P0_hat_avg).tolist()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-path",
        default="../../../constrained-ssm-internal-asu-advi-fix-input/synthetic_data_files/standalone_external_bundle.npz",
    )
    parser.add_argument("--result-root", default="external_notebook_runs")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-val", type=int, default=64)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument("--history-lags", type=int, default=3)
    parser.add_argument("--weight-clip", type=float, default=10.0)
    parser.add_argument("--likelihood-loss", choices=["factorized", "exact_pairwise"], default="exact_pairwise")
    parser.add_argument("--propensity-model", choices=["logistic", "lstm"], default="lstm")
    parser.add_argument("--propensity-lstm-hidden-dim", type=int, default=32)
    parser.add_argument("--propensity-lstm-num-layers", type=int, default=1)
    parser.add_argument("--propensity-lstm-lr", type=float, default=1e-3)
    parser.add_argument("--propensity-lstm-epochs", type=int, default=100)
    args = parser.parse_args()

    result_root = args.result_root
    os.makedirs(result_root, exist_ok=True)

    bundle = np.load(args.bundle_path, allow_pickle=True)

    obs = bundle["obs"]
    inputs = bundle["inputs"]
    states = bundle["states"]
    labels = bundle["labels"]

    K_true = int(bundle["K_true"])
    D = int(bundle["D"])
    M = int(bundle["M"])
    T = int(bundle["T"])

    obs_split, tags_split, inputs_split, truth = split_external_data(
        obs=obs,
        inputs=inputs,
        labels=labels,
        states=states,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
    )

    # monkeypatch config to match notebook dimensions
    sc.make_config = make_external_config_factory(obs, inputs, states, labels, K_true, D, M, T)

    treatment_split = tuple(x[..., 0] for x in inputs_split)
    propensity_extra_split = tuple(x[..., 1:] for x in inputs_split)
    treat_only_inputs_split = tuple(x[..., :1] for x in inputs_split)
    full_inputs_split = inputs_split

    variants = [
        ("treatment_only", treat_only_inputs_split, 1, False, "treatment_only"),
        #("full_inputs", full_inputs_split, M, False, "full_inputs"),
        ("treatment_only_history_weighted", treat_only_inputs_split, 1, True, "treatment_only"),
        ("full_inputs_history_weighted", full_inputs_split, M, True, "full_inputs"),
    ]

    metrics = {}
    for name, hmm_inputs_split, m_dim, weighted, observed_mode in variants:
        run_dir = os.path.join(result_root, name)

        # fail early on wiring errors
        dry_run_shape_check(
            obs_split=obs_split,
            treatment_split=treatment_split,
            hmm_inputs_split=hmm_inputs_split,
            full_inputs_split=full_inputs_split,
            bundle=bundle,
            m_dim=m_dim,
            observed_mode=observed_mode,
        )

        advi_hmm, save_file, C, weight_summary, cluster_file = sc.fit_model(
            obs_split=obs_split,
            tags_split=tags_split,
            treatment_split=treatment_split,
            hmm_inputs_split=hmm_inputs_split,
            result_dir=run_dir,
            M=m_dim,
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
            num_states=K_true,
        )

        metrics[name] = evaluate_external_bundle(
            standalone_module=sc,
            advi_hmm=advi_hmm,
            save_file=save_file,
            cluster_file=cluster_file,
            C=C,
            obs_split=obs_split,
            hmm_inputs_split=hmm_inputs_split,
            full_inputs_split=full_inputs_split,
            truth=truth,
            bundle=bundle,
            observed_mode=observed_mode,
        )
        metrics[name]["likelihood_loss"] = args.likelihood_loss
        if weighted:
            metrics[name]["propensity_model"] = args.propensity_model
        if weight_summary is not None:
            metrics[name].update(weight_summary)

        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(metrics[name], f, indent=2)

    with open(os.path.join(result_root, "comparison.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
