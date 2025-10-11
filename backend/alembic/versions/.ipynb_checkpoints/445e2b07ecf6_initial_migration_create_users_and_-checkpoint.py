"""Initial migration: create users and segmentation_history tables

Revision ID: 445e2b07ecf6
Revises: 
Create Date: 2025-03-06 09:00:02.617039

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '445e2b07ecf6'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
