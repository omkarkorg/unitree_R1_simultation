import gymnasium as gym

TASK = "R1-WholeBody-Flat"
TASK_PLAY = "R1-WholeBody-Flat-Play"

gym.register(
    id=TASK,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:R1WholeBodyEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:R1WholeBodyPPORunnerCfg",
            "default_agent": "rsl_rl"},
)
gym.register(
    id=TASK_PLAY,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:R1WholeBodyEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:R1WholeBodyPPORunnerCfg",
            "default_agent": "rsl_rl"},
)
