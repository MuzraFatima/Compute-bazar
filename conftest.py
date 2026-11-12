"""
conftest.py — Pytest configuration for Compute Bazaar tests.
Adds the project root to sys.path so imports like
`from backend.app.auction.engine import ...` work correctly.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
