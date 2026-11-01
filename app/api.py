"""Generates JSON endpoints for the campaign economics simulator."""

from typing import Any

from flask import abort, Blueprint, current_app, request

from app.helpers import compute_economics

bp = Blueprint("api", __name__)


@bp.route("/economics", methods=["POST"])
def economics() -> tuple[dict[str, Any], int]:
    """Recomputes campaign economics under the posted assumptions."""
    try:
        rendered = compute_economics(
            request.get_json(),
            current_app.config["Y_TRUE"],
            current_app.config["Y_PROBA"],
        )
    except (ValueError, TypeError, KeyError) as error:
        abort(400, description=str(error))

    return rendered, 200
