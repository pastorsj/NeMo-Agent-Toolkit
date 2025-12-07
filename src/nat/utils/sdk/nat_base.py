from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import ClassVar
from typing import TypeVar
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import model_validator

T = TypeVar("T")

logger = logging.getLogger(__name__)


class NatBase(BaseModel):
    """Base class for all NAT SDK components.

    Subclasses must set the `_marker_class` class variable to specify their base config type.
    This enables the `computed_name` property to automatically determine the correct name.

    If `registered_function` is provided, the component will be automatically registered
    with the GlobalTypeRegistry for local development/prototyping. Note that for production
    use, you should use the appropriate `@register_*` decorator in a Python file.
    """

    # Subclasses must override this with their specific base config class
    # e.g., NatLLM sets _marker_class = LLMBaseConfig
    _marker_class: ClassVar[type] = None  # type: ignore

    name: str | None = Field(description="Name of the NAT component.", exclude=True, default=None)
    registered_function: Callable | None = Field(
        description=("An async generator function that yields the built component. "
                     "For development/prototyping only - use @register_* decorators for production."),
        default=None,
        exclude=True,
    )
    # Internal attribute to hold the computed name in case the user does not provide one.
    _computed_name: str | None = PrivateAttr(default=None)
    # Track if we registered this function dynamically
    _dynamically_registered: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _register_function_if_needed(self) -> NatBase:
        """
        If registered_function is provided, automatically register it with the GlobalTypeRegistry.

        This enables rapid prototyping in notebooks without needing to create separate files
        with @register_* decorators. However, for production use, the proper decorators
        should be used.
        """
        if self.registered_function is None:
            return self

        if self._marker_class is None:
            logger.warning(
                "%s has registered_function but no _marker_class set. Cannot auto-register.",
                self.__class__.__name__,
            )
            return self

        # Get the config type from the class hierarchy
        config_type = self._get_config_type_for_registration()
        if config_type is None:
            logger.warning(
                "Could not determine config type for %s. Cannot auto-register.",
                self.__class__.__name__,
            )
            return self

        # Check if already registered
        if self._is_already_registered(config_type):
            return self

        # Register the function
        self._do_register(config_type, self.registered_function)
        self._dynamically_registered = True

        # Warn about production usage
        warnings.warn(
            f"Component '{self.__class__.__name__}' was dynamically registered for development. "
            f"For production, create a Python file with the appropriate @register_* decorator. "
            f"See the documentation for the correct decorator pattern.",
            UserWarning,
            stacklevel=4,
        )

        return self

    def _get_config_type_for_registration(self) -> type | None:
        """Get the config type that should be used for registration."""
        # Find the config class in the MRO that directly inherits from the marker class
        mro = self.__class__.mro()

        for cls in mro:
            if cls is self.__class__:
                continue
            if cls is self._marker_class:
                continue
            if cls is NatBase:
                continue
            if cls is BaseModel:
                continue

            # Check if this class inherits from the marker class
            try:
                if issubclass(cls, self._marker_class) and hasattr(cls, 'full_type'):
                    return cls
            except TypeError:
                continue

        return None

    def _is_already_registered(self, config_type: type) -> bool:
        """Check if this config type is already registered in the GlobalTypeRegistry."""
        from nat.cli.type_registry import GlobalTypeRegistry
        from nat.data_models.authentication import AuthProviderBaseConfig
        from nat.data_models.embedder import EmbedderBaseConfig
        from nat.data_models.evaluator import EvaluatorBaseConfig
        from nat.data_models.front_end import FrontEndBaseConfig
        from nat.data_models.function import FunctionBaseConfig
        from nat.data_models.function import FunctionGroupBaseConfig
        from nat.data_models.llm import LLMBaseConfig
        from nat.data_models.logging import LoggingBaseConfig
        from nat.data_models.memory import MemoryBaseConfig
        from nat.data_models.middleware import MiddlewareBaseConfig
        from nat.data_models.object_store import ObjectStoreBaseConfig
        from nat.data_models.retriever import RetrieverBaseConfig
        from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig

        registry = GlobalTypeRegistry.get()

        # Check each registry based on the marker class
        # Note: We access protected members intentionally to check registration status
        # pylint: disable=protected-access
        try:
            if issubclass(self._marker_class, FunctionBaseConfig):
                return config_type in registry._registered_functions  # noqa: SLF001
            elif issubclass(self._marker_class, FunctionGroupBaseConfig):
                return config_type in registry._registered_function_groups  # noqa: SLF001
            elif issubclass(self._marker_class, LLMBaseConfig):
                return config_type in registry._registered_llm_provider_infos  # noqa: SLF001
            elif issubclass(self._marker_class, EmbedderBaseConfig):
                return config_type in registry._registered_embedder_provider_infos  # noqa: SLF001
            elif issubclass(self._marker_class, RetrieverBaseConfig):
                return config_type in registry._registered_retriever_provider_infos  # noqa: SLF001
            elif issubclass(self._marker_class, MemoryBaseConfig):
                return config_type in registry._registered_memory_infos  # noqa: SLF001
            elif issubclass(self._marker_class, EvaluatorBaseConfig):
                return config_type in registry._registered_evaluator_infos  # noqa: SLF001
            elif issubclass(self._marker_class, MiddlewareBaseConfig):
                return config_type in registry._registered_middleware  # noqa: SLF001
            elif issubclass(self._marker_class, AuthProviderBaseConfig):
                return config_type in registry._registered_auth_provider_infos  # noqa: SLF001
            elif issubclass(self._marker_class, ObjectStoreBaseConfig):
                return config_type in registry._registered_object_store_infos  # noqa: SLF001
            elif issubclass(self._marker_class, TelemetryExporterBaseConfig):
                return config_type in registry._registered_telemetry_exporters  # noqa: SLF001
            elif issubclass(self._marker_class, LoggingBaseConfig):
                return config_type in registry._registered_logging_methods  # noqa: SLF001
            elif issubclass(self._marker_class, FrontEndBaseConfig):
                return config_type in registry._registered_front_end_infos  # noqa: SLF001
        except TypeError:
            pass

        return False

    def _do_register(self, config_type: type, build_fn: Callable) -> None:
        """Register the component by calling the appropriate decorator from register_workflow.

        This reuses the existing decorator functions, just calling them programmatically
        instead of using them as decorators.
        """
        from nat.cli.register_workflow import register_auth_provider
        from nat.cli.register_workflow import register_embedder_provider
        from nat.cli.register_workflow import register_evaluator
        from nat.cli.register_workflow import register_front_end
        from nat.cli.register_workflow import register_function
        from nat.cli.register_workflow import register_function_group
        from nat.cli.register_workflow import register_llm_provider
        from nat.cli.register_workflow import register_logging_method
        from nat.cli.register_workflow import register_memory
        from nat.cli.register_workflow import register_middleware
        from nat.cli.register_workflow import register_object_store
        from nat.cli.register_workflow import register_retriever_provider
        from nat.cli.register_workflow import register_telemetry_exporter
        from nat.data_models.authentication import AuthProviderBaseConfig
        from nat.data_models.embedder import EmbedderBaseConfig
        from nat.data_models.evaluator import EvaluatorBaseConfig
        from nat.data_models.front_end import FrontEndBaseConfig
        from nat.data_models.function import FunctionBaseConfig
        from nat.data_models.function import FunctionGroupBaseConfig
        from nat.data_models.llm import LLMBaseConfig
        from nat.data_models.logging import LoggingBaseConfig
        from nat.data_models.memory import MemoryBaseConfig
        from nat.data_models.middleware import MiddlewareBaseConfig
        from nat.data_models.object_store import ObjectStoreBaseConfig
        from nat.data_models.retriever import RetrieverBaseConfig
        from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig

        try:
            # Call the appropriate decorator function programmatically
            # The decorator pattern is: register_*(config_type)(build_fn)
            if issubclass(self._marker_class, FunctionBaseConfig):
                register_function(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered function: %s", config_type.full_type)

            elif issubclass(self._marker_class, FunctionGroupBaseConfig):
                register_function_group(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered function group: %s", config_type.full_type)

            elif issubclass(self._marker_class, LLMBaseConfig):
                register_llm_provider(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered LLM provider: %s", config_type.full_type)

            elif issubclass(self._marker_class, EmbedderBaseConfig):
                register_embedder_provider(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered embedder provider: %s", config_type.full_type)

            elif issubclass(self._marker_class, RetrieverBaseConfig):
                register_retriever_provider(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered retriever provider: %s", config_type.full_type)

            elif issubclass(self._marker_class, MemoryBaseConfig):
                register_memory(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered memory: %s", config_type.full_type)

            elif issubclass(self._marker_class, EvaluatorBaseConfig):
                register_evaluator(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered evaluator: %s", config_type.full_type)

            elif issubclass(self._marker_class, MiddlewareBaseConfig):
                register_middleware(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered middleware: %s", config_type.full_type)

            elif issubclass(self._marker_class, AuthProviderBaseConfig):
                register_auth_provider(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered auth provider: %s", config_type.full_type)

            elif issubclass(self._marker_class, ObjectStoreBaseConfig):
                register_object_store(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered object store: %s", config_type.full_type)

            elif issubclass(self._marker_class, TelemetryExporterBaseConfig):
                register_telemetry_exporter(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered telemetry exporter: %s", config_type.full_type)

            elif issubclass(self._marker_class, LoggingBaseConfig):
                register_logging_method(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered logging method: %s", config_type.full_type)

            elif issubclass(self._marker_class, FrontEndBaseConfig):
                register_front_end(config_type=config_type)(build_fn)
                logger.debug("Dynamically registered front end: %s", config_type.full_type)

            else:
                logger.warning(
                    "Unknown marker class %s for %s. Cannot auto-register.",
                    self._marker_class.__name__,
                    self.__class__.__name__,
                )

        except Exception as e:
            logger.error("Failed to register %s: %s", self.__class__.__name__, e)
            raise

    @property
    def computed_name(self) -> str:
        """Get the computed name for this component using its marker class.

        Returns:
            str: The computed name for this component.

        Raises:
            NotImplementedError: If the subclass hasn't set _marker_class.
        """
        if self._marker_class is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set _marker_class class variable to use computed_name property")
        return self._compute_name(self._marker_class)

    @property
    def computed_config(self):
        """Get the computed config for this component using its marker class.

        Returns:
            The computed config object for this component.

        Raises:
            NotImplementedError: If the subclass hasn't set _marker_class.
        """
        if self._marker_class is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set _marker_class class variable to use computed_config property")
        return self._compute_config(self._marker_class)

    def _compute_name(self, marker_cls: type[T]) -> str:
        """Compute the name for this component using the specified marker class."""
        name, _ = self.compute_name_and_config(marker_cls)
        return name

    def _compute_config(self, marker_cls: type[T]) -> T:
        """Compute the config for this component using the specified marker class."""
        _, config = self.compute_name_and_config(marker_cls)
        return config

    def compute_name_and_config(self, marker_cls: type[T]) -> tuple[str, T]:
        """Compute both name and config for this component using the specified marker class.

        This is the primary method for extracting the name and configuration from a NAT component.
        It handles name generation and casts the component to its base ancestor config type.

        Args:
            marker_cls: The base config class to cast to (e.g., LLMBaseConfig, FunctionBaseConfig)

        Returns:
            A tuple of (name, config) where name is the component name and config is the
            casted configuration object.
        """
        config_type = self._cast_to_base_ancestor(marker_cls)

        name = self.name

        if name is not None and name != "":
            return name, config_type
        elif self._computed_name is not None and self._computed_name != "":
            return self._computed_name, config_type
        else:
            component_name = getattr(config_type, 'type', self.__class__.__name__)
            # Create unique component name
            computed_name = f"{component_name}_{uuid4().hex}"

            # Key to this working. If the name does not initially exist, we set it here so that subsequent calls
            # to compute_name will return the same name for the same instance.

            self._computed_name = computed_name

            return computed_name, config_type

    def _cast_to_base_ancestor(self, marker_cls: type[T]) -> T:
        """
        Finds the specific ancestor class in the hierarchy that inherits from
        'marker_cls' but is closest to it (the 'root' of that branch).
        """
        # 1. Get the linear history of the class (e.g., [Admin, SuperUser, User, Person, ...])
        mro = self.__class__.mro()

        # 2. Filter the list:
        #    - Must be a subclass of the marker (Person)
        #    - Must NOT be the marker itself (we want User, not Person)
        candidates = [
            cls for cls in mro if issubclass(cls, marker_cls) and cls is not marker_cls and cls is not self.__class__
        ]

        if not candidates:
            raise TypeError(f"No ancestor of {self.__class__.__name__} inherits from {marker_cls.__name__}")

        # 3. Pick the FIRST candidate.
        #    In the MRO, classes closer to the current class appear first.
        #    We want the most specific config class (e.g., ReActAgentWorkflowConfig),
        #    not a more general one (e.g., NatAgent or AgentBaseConfig).
        target_cls = candidates[0]

        # 4. Create instance with all fields (so validation passes), but preserve
        #    which fields were explicitly set so config file serialization only
        #    includes those fields (via exclude_unset=True).
        #    The 'name' and 'registered_function' fields are already excluded via
        #    Field(exclude=True) in their definitions.
        instance = target_cls(**self.model_dump())

        # 5. Preserve the original model_fields_set so that when the config is
        #    serialized with exclude_unset=True, only explicitly set fields are included.
        #    Filter to only include fields that exist in the target class.
        target_field_names = set(target_cls.model_fields.keys())  # type: ignore[attr-defined]
        preserved_fields_set = self.model_fields_set & target_field_names
        instance.__pydantic_fields_set__.clear()  # type: ignore[union-attr]
        instance.__pydantic_fields_set__.update(preserved_fields_set)  # type: ignore[union-attr]

        return instance
