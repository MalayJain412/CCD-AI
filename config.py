# config.py

import os
from urllib.parse import quote_plus

class Config:
    """
    Base configuration settings for the Flask application.
    """
    # It's good practice to have a secret key, even if not used yet.
    SECRET_KEY = os.getenv('SECRET_KEY', 'a-very-secret-key')
    
    # SQLAlchemy configuration
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # --- Azure SQL Database Connection String ---
    # This is the connection string for your database.
    
    DB_USER = "Malay"
    DB_PASS = "Greenscan@12"
    DB_SERVER = "greenscanserver.database.windows.net"
    DB_NAME = "chatbot_db"
    
    # URL encode the password to handle special characters like '@'
    encoded_pass = quote_plus(DB_PASS)

    # The final, correctly formatted and encoded connection string
    SQLALCHEMY_DATABASE_URI = (
        f"mssql+pyodbc://{DB_USER}:{encoded_pass}@{DB_SERVER}:1433/{DB_NAME}"
        "?driver=ODBC+Driver+18+for+SQL+Server"
        "&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30"
    )

