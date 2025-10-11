"""create_users_and_segmentation_history_tables

Revision ID: 1da915619096
Revises: dfbcaf0e4f0c
Create Date: 2025-03-06 08:56:19.902230

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1da915619096'
down_revision: Union[str, None] = 'dfbcaf0e4f0c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
