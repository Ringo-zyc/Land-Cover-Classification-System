"""Initial migration: create users and segmentation_history tables

Revision ID: 1da915619096
Revises: 
Create Date: 2025-03-06 09:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# 确保 revision 与文件名中的 ID 一致
revision = '1da915619096'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # 创建 users 表
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(), nullable=True),
        sa.Column('password_hash', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_users_id', 'id'),
        sa.Index('ix_users_username', 'username', unique=True)
    )
    # 创建 segmentation_history 表
    op.create_table(
        'segmentation_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('image_id', sa.String(), nullable=False),
        sa.Column('segmented_url', sa.String(), nullable=False),
        sa.Column('stats', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_segmentation_history_user_id', 'user_id'),
        sa.Index('ix_segmentation_history_created_at', 'created_at'),
        sa.UniqueConstraint('user_id', 'image_id', name='unique_user_image')
    )

def downgrade():
    op.drop_table('segmentation_history')
    op.drop_table('users')