"""The class contract every registered extension (rule, source, component) meets.

Registries check it before they change anything, so an invalid implementation is
rejected at registration with its name and the violated contract, instead of
failing later with a raw ``AttributeError`` when it is used.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ExtensionContract:
    """What a registry requires of the classes it registers.

    ``base`` is the class they must subclass, ``identifier`` the class attribute
    holding their nonempty string id and ``label`` how messages name that id.
    ``metadata`` lists further class attributes that must be set (truthy).
    """

    kind: str
    base: type
    identifier: str
    label: str = "name"
    metadata: tuple[str, ...] = ()

    def check(self, component: object) -> str:
        """The component's id, once it meets the contract.

        Raises:
            TypeError: if it is not a class, does not subclass ``base``, is abstract,
                lacks metadata, has no nonempty string id or its ``Params`` is not a pydantic model.
        """
        if not (inspect.isclass(component) and issubclass(component, self.base)):
            shown = getattr(component, "__name__", type(component).__name__)
            raise TypeError(f"{shown}: a {self.kind} must subclass {self.base.__name__}")
        if inspect.isabstract(component):
            raise TypeError(f"{component.__name__} is abstract and cannot be registered")
        if missing := [name for name in self.metadata if not getattr(component, name, None)]:
            raise TypeError(
                f"{component.__name__} is missing {self.kind} metadata: {', '.join(missing)}"
            )
        identifier = getattr(component, self.identifier, None)
        if not isinstance(identifier, str) or not identifier:
            raise TypeError(f"{component.__name__} has no {self.kind} {self.label}")
        params = getattr(component, "Params", None)
        if not (isinstance(params, type) and issubclass(params, BaseModel)):
            raise TypeError(f"{component.__name__}.Params must be a pydantic model")
        return identifier


__all__ = ["ExtensionContract"]
