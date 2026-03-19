#!/usr/bin/env python3
import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device

EXPERIMENTS = [
    ('shared', 'shared', 3),
    ('segment3', 'segment', 3),
]


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s %(message)s')
    checkpoint_dir = Path('/mnt/sda/xzh/lerobot_outputs/pi0_residual_everystep_alpha1_20260317_015201/checkpoints/050000/pretrained_model')
    output_root = Path('/mnt/sda/xzh/lerobot_outputs/evals')

    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ['LEROBOT_FORCE_TASK_TEXT'] = 'Push the T shaped block to the target.'

    seed = 1000
    set_seed(seed)

    env_cfg = make_env_config('pusht')
    envs = make_env(env_cfg, n_envs=10, use_async_envs=False)

    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
    policy_cfg.pretrained_path = checkpoint_dir
    policy_cfg.device = 'cuda'
    policy_cfg.use_amp = False

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()

    preprocessor_overrides = {
        'device_processor': {'device': str(policy.config.device)},
        'rename_observations_processor': {'rename_map': {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    device = get_safe_torch_device(policy.config.device, log=True)
    summary = []

    for tag, mode, num_segments in EXPERIMENTS:
        out_dir = output_root / f'pi0_residual_everystep_{tag}_ckpt050000_ep50'
        out_dir.mkdir(parents=True, exist_ok=True)

        policy.config.inference_base_alpha_override = 1.5
        policy.config.use_stepwise_base_gate = False
        policy.config.base_gate_values = ()
        policy.config.base_postprocess_mode = mode
        policy.config.base_num_segments = num_segments
        if hasattr(policy, 'model') and hasattr(policy.model, 'config'):
            policy.model.config.inference_base_alpha_override = 1.5
            policy.model.config.use_stepwise_base_gate = False
            policy.model.config.base_gate_values = ()
            policy.model.config.base_postprocess_mode = mode
            policy.model.config.base_num_segments = num_segments

        logging.info('start %s mode=%s num_segments=%s', tag, mode, num_segments)
        with torch.no_grad(), torch.autocast(device_type=device.type) if policy.config.use_amp else nullcontext():
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=50,
                max_episodes_rendered=10,
                videos_dir=out_dir / 'videos',
                start_seed=seed,
                max_parallel_tasks=env_cfg.max_parallel_tasks,
            )

        info['experiment_tag'] = tag
        info['inference_base_alpha_override'] = 1.5
        info['base_postprocess_mode'] = mode
        info['base_num_segments'] = num_segments
        with open(out_dir / 'eval_info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

        overall = info['overall']
        summary.append({
            'tag': tag,
            'inference_base_alpha_override': 1.5,
            'base_postprocess_mode': mode,
            'base_num_segments': num_segments,
            'avg_sum_reward': overall['avg_sum_reward'],
            'avg_max_reward': overall['avg_max_reward'],
            'pc_success': overall['pc_success'],
            'n_episodes': overall['n_episodes'],
            'out_dir': str(out_dir),
        })
        with open(output_root / 'pi0_residual_everystep_postprocessAB_ckpt050000_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logging.info('done %s overall=%s', tag, overall)

    from lerobot.envs.utils import close_envs
    close_envs(envs)
    logging.info('postprocess AB eval finished')


if __name__ == '__main__':
    main()
