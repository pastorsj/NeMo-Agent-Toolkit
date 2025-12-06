from collections.abc import Callable
from typing import ClassVar
from typing import TypeVar
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr

T = TypeVar("T")


class NatBase(BaseModel):
    """Base class for all NAT SDK components.

    Subclasses must set the `_marker_class` class variable to specify their base config type.
    This enables the `computed_name` property to automatically determine the correct name.
    """
    """Base class for all NAT SDK components.

    Subclasses must set the `_marker_class` class variable to specify their base config type.
    This enables the `computed_name` property to automatically determine the correct name.
    """

    # Subclasses must override this with their specific base config class
    # e.g., NatLLM sets _marker_class = LLMBaseConfig
    _marker_class: ClassVar[type] = None  # type: ignore

    name: str | None = Field(description="Name of the NAT component.", exclude=True, default=None)
    registered_function: Callable | None = Field(
        description="Registered function for the NAT component.",
        default=None,
        exclude=True,
    )
    # Internal attribute to hold the computed name in case the user does not provide one.
    _computed_name: str | None = PrivateAttr(default=None)

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
