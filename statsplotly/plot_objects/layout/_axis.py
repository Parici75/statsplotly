from __future__ import annotations

import datetime
import logging
from typing import Any, TypeAlias

from pydantic import ValidationInfo, field_validator, model_validator

from statsplotly._base import BaseModel

axis_coordinates_type: TypeAlias = float | datetime.datetime | str

logger = logging.getLogger(__name__)


class ColorAxis(BaseModel):
    cmin: float | None = None
    cmax: float | None = None
    colorbar: dict[str, Any] | None = None
    colorscale: str | list[list[float | str]] | None = None
    showscale: bool | None = None


class BaseAxis(BaseModel):
    """Compatible properties with 2D and 3D (Scene) Layout."""

    title: str | None = None
    range: list[axis_coordinates_type] | None = None  # noqa: A003
    type: str | None = None  # noqa: A003
    autorange: bool | str | None = None
    showgrid: bool | None = None
    tickmode: str | None = None
    tickvals: list[axis_coordinates_type] | None = None
    ticktext: list[str] | None = None
    zeroline: bool | None = None

    @field_validator("autorange")
    def validate_autorange(
        cls, value: bool | str | None, info: ValidationInfo
    ) -> bool | str | None:
        if info.data.get("range") is not None:
            return False
        return value

    @model_validator(mode="after")  # type: ignore
    def validate_axis_consistency(cls, model: BaseAxis) -> BaseAxis:
        if model.range is not None and model.tickvals is not None:
            model.tickvals = model.range
            if model.ticktext is not None:
                try:
                    model.ticktext = [
                        model.ticktext[int(idx)] for idx in model.tickvals  # type: ignore
                    ]
                except TypeError:
                    logger.error("Can not adjust tick text labels to tick values")

        return model


class XYAxis(BaseAxis):
    automargin: bool = True
    scaleanchor: str | None = None
    scaleratio: float | None = None
    constrain: str | None = None
