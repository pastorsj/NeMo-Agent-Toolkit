from collections.abc import Callable

from pydantic import BaseModel
from pydantic import Field


class NatBase(BaseModel):
    name: str = Field(description="Name of the NAT component.", exclude=True, default="")
    registered_function: Callable | None = Field(
        description="Registered function for the NAT component.",
        default=None,
        exclude=True,
    )

    def investigate(self) -> None:
        print(self.__dict__)
