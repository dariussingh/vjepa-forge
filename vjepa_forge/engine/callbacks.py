class CallbackRegistry:
    def __init__(self) -> None:
        self.callbacks: dict[str, list] = {}

    def register(self, name: str, fn) -> None:
        self.callbacks.setdefault(name, []).append(fn)

    def run(self, name: str, *args, **kwargs) -> None:
        for fn in self.callbacks.get(name, []):
            fn(*args, **kwargs)
