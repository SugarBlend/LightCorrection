import torch
import mlflow
import logging
import clearml
import argparse
import torchinfo
from pathlib import Path
from typing import Tuple, Union
from tempfile import TemporaryDirectory
import core.gan_trainer
import core.simple_trainer
from utils.parser import ExperimentParser
from core.evaluation import Evaluator

correspondence = {
    'gan': core.gan_trainer.GANTrainer,
    'standard': core.simple_trainer.SimpleTrainer
}


def prepare_logging(
        args: argparse.Namespace,
) -> Tuple[Union[clearml.Task, mlflow.ActiveRun], logging.Logger]:
    if args.logging["backend"] == "mlflow":
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment(args.logging["project_name"])
        mlflow.start_run(run_name=args.logging["task_name"], log_system_metrics=True)
        active_logger = mlflow.active_run()
        with TemporaryDirectory() as tempdir:
            for model_name, config in config_parser.model_config.items():
                mlflow.log_dict(config, artifact_file=f"configs/{model_name}.json")
            mlflow.log_dict(config_parser.experiment_config, artifact_file="configs/experiment.json")
            mlflow.log_dict(config_parser.datasets_config, artifact_file="configs/dataset_config.json")
            for model_name in model:
                with open(f'{tempdir}/{model_name}_summary.txt', "w", encoding="utf-8") as file:
                    file.write(str(torchinfo.summary(model[model_name])))
                mlflow.log_artifact(f'{tempdir}/{model_name}_summary.txt', artifact_path='net_description')

    elif args.logging["backend"] == "clearml":
        active_logger = clearml.Task.init(project_name=args.logging["project_name"], task_name=args.logging["task_name"])
    else:
        raise NotImplementedError

    logging.basicConfig(level=logging.INFO)
    name = args.logging.get("backend")
    logger = logging.getLogger(name)

    return active_logger, logger


if __name__ == '__main__':
    # config_parser = ExperimentParser("configs/denoise_experiments.yaml")
    config_parser = ExperimentParser("configs/low_light_enhancement_experiments.yaml")
    mode = config_parser.experiment_config['training']['mode']
    TrainerClass = correspondence[mode]

    args = argparse.Namespace(**{"training": config_parser.experiment_config["training"]},
                              **{"logging": config_parser.experiment_config["logging"]},
                              **{"datasets": config_parser.datasets_config})

    while True:
        model = config_parser.get_model()
        if model is None:
            break

        optimizer = config_parser.get_optimizer(model)
        scheduler = config_parser.get_scheduler(optimizer)
        loss_function = config_parser.get_loss_function()
        train_loader, test_loader = config_parser.get_dataloaders().values()

        active_logger, logger = prepare_logging(args)

        trainer = TrainerClass(args, active_logger, logger, model, optimizer, scheduler, loss_function, train_loader,
                               test_loader)
        best_model_path = trainer.run()

        if args.logging["backend"] == "clearml":
            artifact = active_logger.artifacts.get(best_model_path)
            artifact_local_path = artifact.get_local_copy()
            trainer.model.load_state_dict(torch.load(artifact_local_path)['model_state_dict'])
        elif args.logging["backend"] == "mlflow":
            uri = f"runs:/{active_logger.info.run_id}/checkpoints/{Path(best_model_path).stem}"
            trainer.model = mlflow.pytorch.load_model(uri)
        trainer.model.eval().cuda()

        evaluator = Evaluator(args, config_parser.datasets['test'], active_logger)
        evaluator.evaluate(trainer.model, args.training['validation'].get("std_range", None))

        if args.logging["backend"] == "mlflow":
            mlflow.end_run()
        elif args.logging["backend"] == "clearml":
            active_logger.close()
