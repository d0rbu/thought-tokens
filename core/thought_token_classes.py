from typing import Self, Any, Type, TypeVar, Protocol, runtime_checkable
from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoTokenizer, AutoConfig
from logger import logger


@runtime_checkable
class ThoughtTokenConfigProtocol[T: PretrainedConfig](Protocol[T]):
    thought_vocab_size: int

    @classmethod
    def from_non_thought_config(cls: Type[Self], config: T, thought_vocab_size: int) -> Self:
        ...


def thought_token_config_class[T: PretrainedConfig](config_class: Type[T]) -> Type[ThoughtTokenConfigProtocol[T]]:
    if issubclass(config_class, ThoughtTokenConfigProtocol):
        logger.warning(f"thought_token_config_class called with a config class that already seems to implement thought token functionality: {config_class}")
        return config_class

    class thought_token_config(config_class):
        def __init__(self: Self, thought_vocab_size: int, **kwargs: Any) -> None:
            super().__init__(**kwargs)

            self.thought_vocab_size = thought_vocab_size
            self.vocab_size += thought_vocab_size

        @classmethod
        def from_non_thought_config(cls: Type['ThoughtTokenConfig'], config: T, thought_vocab_size: int) -> Self:
            return cls(thought_vocab_size, **config.__dict__)

    return thought_token_config


@runtime_checkable
class ThoughtTokenModelProtocol[U: PreTrainedModel](Protocol[U]):
    # TODO: methods that will be useful for thought token models
    ...


def thought_token_model_config_classes[U: PreTrainedModel, T: PretrainedConfig](model_class: Type[U], config_class: Type[ThoughtTokenConfigProtocol[T]] | Type[T] | None = None) -> tuple[Type[ThoughtTokenModelProtocol[U]], Type[ThoughtTokenConfigProtocol[T]]]:
    if config_class is None:
        if hasattr(model_class, 'config_class'):
            config_class: Type[ThoughtTokenConfigProtocol[T]] | Type[T] = model_class.config_class
        else:
            raise ValueError("config_class must be provided if model_class does not have a config_class attribute")

    if issubclass(config_class, ThoughtTokenConfigProtocol):
        new_config_class = config_class
    else:
        new_config_class = thought_token_config_class(config_class)

    if issubclass(model_class, ThoughtTokenModelProtocol):
        logger.warning(f"thought_token_model_config_classes called with a model class that already seems to implement thought token functionality: {model_class}")
        model_config_class = getattr(model_class, 'config_class', None)
        if model_config_class != new_config_class:
            logger.warning(f"config class of thought token model class does not match the config class: {model_config_class} != {new_config_class}")

        thought_token_model = model_class
    else:
        class thought_token_model(model_class):
            config_class = new_config_class

            def __init__(self: Self, config: ThoughtTokenConfigProtocol[PretrainedConfig], **kwargs: Any) -> None:
                super().__init__(config, **kwargs)

    return thought_token_model, new_config_class
