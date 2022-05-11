import argparse

from entity_gym.examples import ENV_REGISTRY
from entity_gym.runner import CliRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MoveToOrigin")
    args = parser.parse_args()

    envs = ENV_REGISTRY
    if args.env not in envs:
        raise ValueError(
            f"Unknown environment {args.env}\nValid environments are {list(envs.keys())}"
        )
    else:
        env_cls = envs[args.env]

    print(env_cls)
    env = env_cls()
    CliRunner(env).run()
