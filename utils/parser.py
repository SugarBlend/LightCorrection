import yaml
import torch
import importlib
import itertools
import albumentations as A
from inspect import getmembers, isclass
from torch.utils.data import DataLoader, Dataset
from typing import Union, List, Dict, Any, Callable, Tuple, Optional


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def dynamic_import(module_name: str, class_name: str) -> Callable:
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _load_custom_transforms(custom_modules: List[str]) -> Dict[str, Callable]:
    transforms: Dict[str, Callable] = {}
    for module_name in custom_modules:
        try:
            module = importlib.import_module(module_name)
            transforms.update({name: obj for name, obj in getmembers(module, isclass)})
        except ModuleNotFoundError:
            print(f"Module {module_name} is not found, skip.")
    return transforms


class ExperimentParser:
    def __init__(self, experiment_path: str) -> None:
        self.experiment_config = load_yaml(experiment_path)
        self.datasets_config = load_yaml(self.experiment_config["datasets"])
        self.training_mode = self.experiment_config["training"]["mode"]

        self.model_config: Dict[str, Any] = {}
        for model_key, model_path in self.experiment_config["training"][self.training_mode].items():
            self.model_config[model_key] = load_yaml(model_path)["model"]

        self.enable_search: Dict[str, bool] = {
            model_key: model_data.get("enable_search", False)
            for model_key, model_data in self.model_config.items()
        }

        self.param_combinations: Dict[str, List[Dict[str, Any]]] = {
            model_key: self._generate_combinations(model_config["arch"]["params"])
            if self.enable_search[model_key] else [model_config["arch"]["params"]]
            for model_key, model_config in self.model_config.items()
        }

        self.current_indices: Dict[str, int] = {key: 0 for key in self.param_combinations}

        self.custom_transforms = _load_custom_transforms(self.datasets_config.get("custom_augmentations", []))
        self.datasets: Optional[Dataset] = None

    def _generate_combinations(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        keys = {key: value for key, value in params.items() if isinstance(value, list)}
        return [dict(zip(keys.keys(), values)) for values in itertools.product(*keys.values())] if keys else [params]

    def get_next_model_config(self) -> Optional[Dict[str, Dict[str, Any]]]:
        next_params = {}
        for model_key, param_list in self.param_combinations.items():
            if self.current_indices[model_key] >= len(param_list):
                return None

            next_params[model_key] = param_list[self.current_indices[model_key]]
            if self.enable_search[model_key]:
                self.current_indices[model_key] += 1

        return next_params

    def get_model(self) -> Optional[Dict[str, torch.nn.Module]]:
        model_dict = {}
        model_params_dict = self.get_next_model_config()
        if model_params_dict is None:
            return None

        for model_name, model_params in model_params_dict.items():
            model_arch = self.model_config[model_name]["arch"]
            ModelClass = dynamic_import(model_arch["module"], model_arch["name"])
            model_dict[model_name] = ModelClass(**model_params)

        return model_dict

    def get_optimizer(self, model_dict: Dict[str, torch.nn.Module]) -> Dict[str, torch.optim.Optimizer]:
        optimizer_dict = {}
        for model_name, model in model_dict.items():
            optimizer_config = self.model_config[model_name]["optimizer"]
            OptimizerClass = dynamic_import(optimizer_config["module"], optimizer_config["name"])
            optimizer_dict[model_name] = OptimizerClass(model.parameters(), **optimizer_config["params"])
        return optimizer_dict

    def get_scheduler(self, optimizer_dict: Dict[str, torch.optim.Optimizer]) -> Dict[str, torch.optim.lr_scheduler.LRScheduler]:
        scheduler_dict = {}
        for model_name, optimizer in optimizer_dict.items():
            scheduler_config = self.model_config[model_name]["scheduler"]
            SchedulerClass = dynamic_import(scheduler_config["module"], scheduler_config["name"])
            scheduler_dict[model_name] = SchedulerClass(optimizer, **scheduler_config["params"])
        return scheduler_dict

    def get_loss_function(self) -> Dict[str, Callable]:
        loss_dict = {}
        for model_name, model_config in self.model_config.items():
            loss_config = model_config["loss"]
            LossClass = dynamic_import(loss_config["module"], loss_config["name"])
            loss_dict[model_name] = LossClass(**loss_config.get("params", {}))
        return loss_dict

    def get_datasets(self) -> Dict[str, Dataset]:
        datasets = {}
        for phase in ["train", "test"]:
            ds_config = self.datasets_config[phase]
            dataset_params = ds_config["dataset"]["params"].copy()

            if "transforms" in dataset_params:
                dataset_params["inputs_transform"], dataset_params["both_transform"] = (
                    self._parse_transforms(dataset_params.pop("transforms"))
                )

            DatasetClass = dynamic_import(ds_config["dataset"]["module"], ds_config["dataset"]["name"])
            datasets[phase] = DatasetClass(**dataset_params)

        return datasets

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        self.datasets = self.get_datasets()
        dataloaders = {}

        for phase, dataset in self.datasets.items():
            dl_config = self.datasets_config[phase]["dataloader"]
            params = dl_config["params"].copy()

            collate_fn = None
            if "collate_fn" in params:
                collate_fn_cfg = params.pop("collate_fn")
                collate_fn = dynamic_import(collate_fn_cfg["module"], collate_fn_cfg["name"])

            DataLoaderClass = dynamic_import(dl_config["module"], dl_config["name"])
            dataloaders[phase] = DataLoaderClass(dataset, collate_fn=collate_fn, **params)

        return dataloaders

    def _parse_transforms(self, transforms_config: Dict[str, Any]) -> Tuple[Optional[Union[A.Compose, A.ReplayCompose]], ...]:
        return tuple(self._create_transform(transforms_config.get(key_type, None)) for key_type in ['inputs', 'both'])

    def _create_transform(self, transform_config: Optional[Dict[str, Any]]) -> Optional[Union[A.Compose, A.ReplayCompose]]:
        if transform_config is None:
            return None

        if not isinstance(transform_config, dict):
            raise ValueError(f"'transform_config' must be a dictionary, received {type(transform_config)}: {transform_config}")

        compose_name = transform_config["name"]
        sub_transforms = transform_config["params"].get("transforms", [])

        transform_instances = []
        for transform in sub_transforms:
            TransformClass = self.custom_transforms.get(transform["name"]) or getattr(A, transform["name"], None)
            if not TransformClass:
                raise ValueError(f"Augmentation not found: '{transform['name']}'!")
            transform_instances.append(TransformClass(**transform.get("params", {})))

        return getattr(A, compose_name)(transform_instances)


if __name__ == '__main__':
    config_parser = ExperimentParser("configs/experiments.yaml")

    while True:
        model_dict = config_parser.get_model()
        if model_dict is None:
            break

        optimizer_dict = config_parser.get_optimizer(model_dict)
        scheduler_dict = config_parser.get_scheduler(optimizer_dict)
        loss_function_dict = config_parser.get_loss_function()
        train_loader, test_loader = config_parser.get_dataloaders().values()
