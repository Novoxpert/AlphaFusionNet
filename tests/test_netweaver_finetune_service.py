"""
test_netweaver_finetune_service.py
Unit tests for netweaver_finetune_service.py
Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Oct 29
"""

import pytest
from unittest.mock import patch, MagicMock
import sys

# --------------------- Test 1: Successful fine-tune flow ---------------------
@patch("subprocess.run")
@patch("sys.exit")
def test_finetune_service_success(mock_exit, mock_run):
    import apps.NetWeaver.src.services.netweaver_finetune_service as nfs

    # Mock subprocess.run always succeeds
    mock_run.return_value = MagicMock(returncode=0)

    # Provide fake command-line arguments
    with patch.object(sys, "argv", ["netweaver_finetune_service.py", "--latest_hours", "2", "--epochs", "5"]):
        nfs.main()

    # sys.exit should not be called on success
    mock_exit.assert_not_called()


# --------------------- Test 2: Pipeline failure triggers sys.exit ---------------------
@patch("subprocess.run")
@patch("sys.exit")
def test_pipeline_failure(mock_exit, mock_run):
    import apps.NetWeaver.src.services.netweaver_finetune_service as nfs

    # Fail the data pipeline step
    def mock_run_side_effect(cmd, check, text, capture_output, cwd):
        if any("data_pipeline" in c for c in cmd):
            raise nfs.subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return MagicMock(returncode=0)

    mock_run.side_effect = mock_run_side_effect

    with patch.object(sys, "argv", ["netweaver_finetune_service.py", "--latest_hours", "2"]):
        nfs.main()
        # Should call sys.exit(1) after pipeline failure
        mock_exit.assert_called_once_with(1)


# --------------------- Test 3: Fine-tune failure triggers sys.exit ---------------------
@patch("subprocess.run")
@patch("sys.exit")
def test_finetune_failure(mock_exit, mock_run):
    import apps.NetWeaver.src.services.netweaver_finetune_service as nfs

    # Fail the finetune step
    def mock_run_side_effect(cmd, check, text, capture_output, cwd):
        if any("train" in c for c in cmd):
            raise nfs.subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return MagicMock(returncode=0)

    mock_run.side_effect = mock_run_side_effect

    with patch.object(sys, "argv", ["netweaver_finetune_service.py", "--latest_hours", "2"]):
        nfs.main()
        # Should call sys.exit(1) after finetune failure
        mock_exit.assert_called_once_with(1)


# --------------------- Test 4: Device and analysis flags ---------------------
@patch("subprocess.run")
@patch("sys.exit")
def test_flags(mock_exit, mock_run):
    import apps.NetWeaver.src.services.netweaver_finetune_service as nfs

    mock_run.return_value = MagicMock(returncode=0)

    with patch.object(sys, "argv", [
        "netweaver_finetune_service.py",
        "--latest_hours", "3",
        "--epochs", "5",
        "--device", "cpu",
        "--no_analysis"
    ]):
        nfs.main()
        mock_exit.assert_not_called()
