from .core import run_experiment, ExperimentConfig


def main():
    config = ExperimentConfig(name="baseline", seed=42, max_steps=5000, render=False)
    run_experiment(config)


if __name__ == "__main__":
    main()
