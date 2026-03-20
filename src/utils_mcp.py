"""
Utility functions for MCP (Model Context Protocol).
"""

from omegaconf import OmegaConf


# =========================
# CONFIG
# =========================
def load_config(path="configs/mcp.yaml"):
    """Load YAML configuration."""
    try:
        return OmegaConf.load(path)
    except Exception as e:
        print(f"❌ Error loading config from {path}: {e}")
        raise
