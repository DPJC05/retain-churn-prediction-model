"""The retain dashboard Flask application factory."""

from datetime import datetime
import json
import sqlite3

from flask import current_app, Flask, g, jsonify, render_template, Response, request
import numpy as np

from src.config import load_config
from src.db import make_connection


def create_app() -> Flask:
    """Flask application factory function. Builds, configures and returns a Flask application instance."""
    app = Flask(__name__)

    config = load_config()
    paths = [
        config.model.metadata_persist_path,
        config.model.curve_data_persist_path,
        config.model.evaluation_report_persist_path,
        config.model.evaluation_arrays_persist_path,
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} was not found. Run `python -m scripts.run_pipeline` first."
            )

    with np.load(config.model.evaluation_arrays_persist_path) as arrays:
        y_true = arrays["y_true"].copy()
        y_proba = arrays["y_proba"].copy()

    app.config.from_mapping(
        # Flask requires a secret key. As the retain dashboard has no sessions, no session cookies are signed.
        SECRET_KEY="dev",  # "dev" is a required placeholder. It is not a secret.
        DB_PATH=config.database.db_path,
        METADATA=json.loads(config.model.metadata_persist_path.read_text()),
        EVALUATION_REPORT=json.loads(
            config.model.evaluation_report_persist_path.read_text()
        ),
        CURVE_DATA=json.loads(config.model.curve_data_persist_path.read_text()),
        Y_TRUE=y_true,
        Y_PROBA=y_proba,
        ECONOMICS=config.economics,
    )
    timestamps = {
        app.config[key].get("trained_at")
        for key in ["METADATA", "EVALUATION_REPORT", "CURVE_DATA"]
    }
    if len(timestamps) > 1:
        app.logger.warning(
            "the metadata, evaluation report and curve data were generated during different model training sessions"
        )

    app.teardown_appcontext(close_db)
    register_filters(app)
    register_error_handlers(app)

    from app.routes import bp as main_bp
    from app.api import bp as api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app


def register_error_handlers(app: Flask) -> None:
    """Attaches the error 404 and 500 handlers."""

    def handle(code: int, message: str):
        def handler(error: Exception) -> tuple[Response | str, int]:
            text = getattr(error, "description", None) or message
            if request.path.startswith("/api/"):
                return jsonify(error=text), code
            return render_template(f"{code}.html", message=text), code

        return handler

    app.register_error_handler(400, handle(400, "Bad Request"))
    app.register_error_handler(404, handle(404, "Resource Not Found"))
    app.register_error_handler(500, handle(500, "Internal Server Error"))


def register_filters(app: Flask) -> None:
    """Registers the Jinja filters that the templates are reliant on."""
    app.jinja_env.filters["currency"] = lambda amount: f"£{amount:,.2f}"
    app.jinja_env.filters["pct"] = lambda proportion: f"{proportion * 100:.1f}%"
    app.jinja_env.filters["prob"] = lambda probability: f"{probability:.2f}"
    app.jinja_env.filters["date"] = format_date


def format_date(iso_string: str | None) -> str:
    """Helper function for register_filters(). Formats an ISO timestamp as a readable date, or a dash if absent."""
    if not iso_string:
        return "—"
    return datetime.fromisoformat(iso_string).strftime("%d %b %Y")


def get_db() -> sqlite3.Connection:
    """Returns the database connection for the current request, opening it if needed."""
    if "db" not in g:
        g.db = make_connection(current_app.config["DB_PATH"])
    return g.db


def close_db(exception: BaseException | None = None) -> None:
    """Closes the database connection. Registered via teardown_appcontext."""
    db = g.pop("db", None)
    if db is not None:
        db.close()
