# strategies/strategy_registry.py
_strategies = {}

def register_strategy(name: str):
    def wrapper(cls):
        _strategies[name] = cls()
        return cls
    return wrapper

def get_strategy(name: str) -> Strategy:
    return _strategies.get(name)

# 使用
@register_strategy("news_broadcast")
class NewsBroadcastStrategy(Strategy):
    ...