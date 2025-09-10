# backend/app/models/user.py
from sqlalchemy import Column, Integer, String, DateTime
from app.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)  # hashed in real app
    role = Column(String(20), default="admin")      # admin / doctor
    created_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"
