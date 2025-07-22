# config.py

import os

class Config:
    """
    Base configuration settings for the Flask application.
    """
    # It's good practice to have a secret key, even if not used yet.
    SECRET_KEY = os.getenv('SECRET_KEY', 'a-very-secret-key')
    
    # SQLAlchemy configuration
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # --- Database URI ---
    # This is the connection string for your database.
    # Replace 'your_password' with the actual password for your 'chatbot_user'.
    # This works for both your local XAMPP and your GCP MySQL server,
    # as they are both running on 'localhost' from the app's perspective.
    SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:@localhost:3307/chatbot_db'
    
    # If you ever want to switch back to the simple file-based database for testing,
    # you can comment out the line above and uncomment the line below.
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///chat_history.db'

