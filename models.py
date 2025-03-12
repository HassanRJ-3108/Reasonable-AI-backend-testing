from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversations = relationship("Conversation", back_populates="user")
    config = relationship("Configuration", back_populates="user", uselist=False)

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"))
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_url = Column(String, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")
    reasoning_steps = relationship("ReasoningStep", back_populates="message", cascade="all, delete-orphan")

class ReasoningStep(Base):
    __tablename__ = "reasoning_steps"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(String, ForeignKey("messages.id"))
    step_number = Column(Integer)
    content = Column(Text)
    type = Column(String)  # "analysis" or "iteration"

    message = relationship("Message", back_populates="reasoning_steps")

class Configuration(Base):
    __tablename__ = "configurations"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    model = Column(String, default="gemini-2.0-flash")
    temperature = Column(Float, default=0.7)
    max_output_tokens = Column(Integer, default=4096)
    enable_reasoning = Column(Boolean, default=True)
    max_iterations = Column(Integer, default=3)
    top_p = Column(Float, default=0.95)
    top_k = Column(Integer, default=64)

    user = relationship("User", back_populates="config")

