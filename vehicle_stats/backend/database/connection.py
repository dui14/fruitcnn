from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "vehicle_stats_db")

class Database:
    client: AsyncIOMotorClient = None
    
    @classmethod
    async def connect_db(cls):
        try:
            cls.client = AsyncIOMotorClient(MONGODB_URL)
            await cls.client.admin.command('ping')
            print(f"Connected to MongoDB at {MONGODB_URL}")
        except ConnectionFailure as e:
            print(f"Could not connect to MongoDB: {e}")
            raise
    
    @classmethod
    async def close_db(cls):
        if cls.client:
            cls.client.close()
            print("MongoDB connection closed")
    
    @classmethod
    def get_database(cls):
        return cls.client[DATABASE_NAME]

async def get_database():
    return Database.get_database()
