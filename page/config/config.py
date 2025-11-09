import os
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY','my_secret') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///data.db')