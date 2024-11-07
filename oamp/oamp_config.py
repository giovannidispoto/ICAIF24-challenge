class ConfigOAMP:
    def __init__(
        self,
        args: dict,
    ):
        self.loss_fn_window = args.get("loss_fn_window", 10)
