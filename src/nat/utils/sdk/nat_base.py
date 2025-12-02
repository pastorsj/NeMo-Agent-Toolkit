from collections.abc import Callable
from typing import TypeVar
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr

T = TypeVar("T")


class NatBase(BaseModel):
    name: str | None = Field(description="Name of the NAT component.", exclude=True, default=None)
    registered_function: Callable | None = Field(
        description="Registered function for the NAT component.",
        default=None,
        exclude=True,
    )
    # Internal attribute to hold the computed name in case the user does not provide one.
    _computed_name: str | None = PrivateAttr(default=None)

    def investigate(self) -> None:
        print(self.__dict__)

    def compute_name(self, marker_cls: T) -> str:
        name, _ = self.compute_name_and_config(marker_cls)
        return name

    def compute_config(self, marker_cls: T) -> T:
        _, config = self.compute_name_and_config(marker_cls)
        return config

    def compute_name_and_config(self, marker_cls: T) -> tuple[str, T]:
        config_type = self.__cast_to_base_ancestor(marker_cls)

        name = self.name

        if name is not None and name != "":
            return name, config_type
        elif self._computed_name is not None and self._computed_name != "":
            return self._computed_name, config_type
        else:
            component_name = config_type.type
            # Create unique component name
            computed_name = f"{component_name}_{uuid4().hex}"

            # Key to this working. If the name does not initially exist, we set it here so that subsequent calls
            # to compute_name will return the same name for the same instance.

            self._computed_name = computed_name

            return computed_name, config_type

    def __cast_to_base_ancestor(self, marker_cls):
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

        # 3. Pick the LAST candidate.
        #    In the MRO, the class "closest" to the marker (User) appears
        #    after the children (Admin, SuperUser).
        target_cls = candidates[-1]

        # 4. Instantiate (Pydantic ignores extra fields by default)
        return target_cls(**self.model_dump(exclude_unset=True))
