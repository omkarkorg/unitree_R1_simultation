from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.environ.get("R1_SIM_DIR", "/workspace/r1_sim"))
from r1_sim import loco_spec as spec


def build_actor(state: dict) -> nn.Sequential:
    linear = sorted({int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight")})
    layers: list[nn.Module] = []
    for n, idx in enumerate(linear):
        w, b = state[f"mlp.{idx}.weight"], state[f"mlp.{idx}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if n < len(linear) - 1:
            if idx + 2 != linear[n + 1]:
                raise RuntimeError(f"unexpected actor layout: {sorted(state)}")
            layers.append(nn.ELU())
    return nn.Sequential(*layers).eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default=os.path.join(os.environ.get("R1_SIM_DIR", "/workspace/r1_sim"),
                                                      "policies", "r1_wholebody.pt"))
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["actor_state_dict"]
    actor = build_actor(state)
    n_in, n_out = actor[0].in_features, actor[-1].out_features
    if (n_in, n_out) != (spec.obs_dim(), len(spec.LEG_JOINTS)):
        raise RuntimeError(f"checkpoint maps {n_in} -> {n_out}, loco_spec expects {spec.obs_dim()} -> {len(spec.LEG_JOINTS)}")

    from rsl_rl.modules import MLP

    hidden = [m.out_features for m in actor if isinstance(m, nn.Linear)][:-1]
    ref = MLP(n_in, n_out, hidden, "elu")
    ref.load_state_dict({k[len("mlp."):]: v for k, v in state.items() if k.startswith("mlp.")})
    ref.eval()
    x = torch.randn(256, n_in) * 2.0
    with torch.no_grad():
        err = (ref(x) - actor(x)).abs().max().item()
    if err > 1e-5:
        raise RuntimeError(f"exported network differs from rsl_rl's MLP by {err}")

    scripted = torch.jit.script(actor)
    meta = {"spec_version": spec.SPEC_VERSION, "obs_dim": n_in, "action_dim": n_out, "hidden": hidden,
            "iteration": int(ckpt.get("iter", -1)), "checkpoint": os.path.abspath(args.checkpoint),
            "trained": datetime.datetime.fromtimestamp(os.path.getmtime(args.checkpoint)).strftime("%Y-%m-%d %H:%M"),
            "action_std": [round(float(v), 4) for v in torch.as_tensor(state.get("distribution.std_param", torch.zeros(0)))]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.jit.save(scripted, args.out, _extra_files={"meta.json": json.dumps(meta)})
    print(f"exported {args.checkpoint} (iteration {meta['iteration']}) -> {args.out}: {n_in} -> {hidden} -> {n_out}, "
          f"max difference to rsl_rl {err:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
