from argparse import Namespace
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from transformers import HfArgumentParser
from ruamel.yaml import YAML

yaml = YAML()

class UMeIParser(HfArgumentParser):
    def __init__(self, *args, use_conf: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_conf = use_conf

    @staticmethod
    def save_args_as_conf(args, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        args_dict = deepcopy(vars(args))
        for k, v in args_dict.items():
            if isinstance(v, Path):
                args_dict[k] = str(v)
        yaml.dump(args_dict, save_path)

    def parse_args_into_dataclasses(self, **kwargs):
        if not self.use_conf:
            return super().parse_args_into_dataclasses(**kwargs)
        from sys import argv
        conf_path = Path(argv[1]).resolve()
        if conf_path.suffix in ['.yml', '.yaml', '.json']:
            conf: dict = yaml.load(conf_path)
        else:
            raise ValueError(f'format not supported for conf: {conf_path.suffix}')
        if conf is None:
            conf = {}

        def to_cli_options(conf: dict[str, Any]) -> list[str]:
            ret = []
            for k, v in conf.items():
                ret.append(f'--{k}')
                if isinstance(v, list):
                    for x in v:
                        ret.append(str(x))
                else:
                    ret.append(str(v))
            return ret

        argv = to_cli_options(conf) + argv[2:]
        # argv = argv[2:]

        for action in self._actions:
            if hasattr(action.type, '__origin__') and issubclass(getattr(action.type, '__origin__'), Iterable):
                action.type = action.type.__args__[0]
                action.nargs = '+'

        args, _ = self.parse_known_args(argv)

        def infer_exp_name() -> Optional[str]:
            if args.conf_root is None:
                return None
            conf_root: Path = args.conf_root.resolve()
            if str(conf_path).startswith(str(conf_root)):
                return str(Path(*conf_path.parts[len(conf_root.parts):-1], conf_path.stem))
            else:
                return None

        if args.exp_name is None:
            args.exp_name = infer_exp_name()

        if args.output_dir is None:
            if args.output_root is None or args.exp_name is None:
                raise AttributeError('unable to infer `output_dir`')
            else:
                args.output_dir = Path(args.output_root) / args.exp_name

        output_dir = Path(args.output_dir)
        self.save_args_as_conf(args, output_dir / 'conf.yml')
        # compatible interface
        return self.parse_dict(vars(args))
