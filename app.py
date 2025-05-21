import os
import logging
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key_for_testing_only")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# Configure the database (PostgreSQL if available, fallback to SQLite)
database_url = os.environ.get("DATABASE_URL")
if database_url is None:
    database_url = "sqlite:///health_detection.db"
    app.logger.warning("DATABASE_URL not found, using SQLite as fallback")
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# This connects the website to the database, so they can talk to each other.
from database import db
db.init_app(app)

from models import HealthRecord, SymptomRecord

with app.app_context():
    db.create_all()

from routes import * 

if __name__ == "__main__":
    app.run(debug=True)