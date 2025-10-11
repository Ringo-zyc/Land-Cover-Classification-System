from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON
from sqlalchemy.sql import func
from sqlalchemy import UniqueConstraint
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    
class SegmentationHistory(Base):
    __tablename__ = "segmentation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # 添加索引
    image_id = Column(String, nullable=False)
    segmented_url = Column(String, nullable=False)
    stats = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)  # 添加索引
    
    __table_args__ = (UniqueConstraint('user_id', 'image_id', name='unique_user_image'),)  # 可选：用户内唯一约束