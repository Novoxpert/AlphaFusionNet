"""
test_netweaver_train_service.py
Description: Unit tests for netweaver_train_service.py
Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Oct 29
"""

import pytest
from unittest.mock import patch, MagicMock
import sys

# --------------------- Test 1: Main flow (success path) ---------------------
@patch("subprocess.run")
def test_main_success(mock_run):
    """Simulate successful training flow."""
    import apps.NetWeaver.src.services.netweaver_train_service as nws

    # Mock subprocess.run → success
    mock_run.return_value = MagicMock(returncode=0)

    # Simulate CLI args
    test_args = [
        "netweaver_train_service.py",
        "--latest_month", "1",
        "--epochs", "10",
        "--model", "CAT"
    ]
    with patch.object(sys, "argv", test_args):
        result = nws.main()
        # The function ends successfully (no sys.exit calls)
        assert result is None
        # Ensure subprocess.run called twice (data + train)
        assert mock_run.call_count == 2


# --------------------- Test 2: Pipeline failure (first subprocess fails) ---------------------
@patch("subprocess.run")
@patch("sys.exit")
def test_pipeline_failure(mock_exit, mock_run):
    """Simulate failure in the data pipeline step."""
    import apps.NetWeaver.src.services.netweaver_train_service as nws

    def mock_run_side_effect(cmd, check, text, capture_output, cwd):
        # Correct detection of the data pipeline command
        if any("data_pipeline" in c for c in cmd):
            raise nws.subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return MagicMock(returncode=0)

    mock_run.side_effect = mock_run_side_effect

    with patch.object(sys, "argv", ["netweaver_train_service.py", "--latest_month", "1"]):
        nws.main()
        mock_exit.assert_called_once_with(1)


# --------------------- Test 3: Model training failure (second subprocess fails) ---------------------
@patch("subprocess.run")
@patch("sys.exit")
def test_training_failure(mock_exit, mock_run):
    """Simulate failure in the model training step."""
    import apps.NetWeaver.src.services.netweaver_train_service as nws

    # Pipeline succeeds, training fails
    def mock_run_side_effect(cmd, check, text, capture_output, cwd):
        if "train" in cmd and "NetWeaver.src.train" in cmd:
            raise nws.subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return MagicMock(returncode=0)
    mock_run.side_effect = mock_run_side_effect

    with patch.object(sys, "argv", ["netweaver_train_service.py", "--latest_month", "1"]):
        nws.main()
        # sys.exit should be called after training failure
        mock_exit.assert_called_once_with(1)


# --------------------- Test 4: CLI argument parsing ---------------------
def test_argparse_basic():
    """Verify that CLI args parse correctly and default values apply."""
    import apps.NetWeaver.src.services.netweaver_train_service as nws

    parser = nws.argparse.ArgumentParser()
    parser.add_argument("--latest_month", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model", type=str, default="CAT")

    args = parser.parse_args(["--latest_month", "3", "--epochs", "50", "--model", "CG"])
    assert args.latest_month == 3
    assert args.epochs == 50
    assert args.model == "CG"


# --------------------- Test 5: run_command helper ---------------------
def test_run_command_success():
    """Ensure run_command returns True on successful subprocess."""
    import apps.NetWeaver.src.services.netweaver_train_service as nws
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        result = nws.run_command(["echo", "hi"], "Test command")
        assert result is True

def test_run_command_failure():
    """Ensure run_command returns False on subprocess failure."""
    import apps.NetWeaver.src.services.netweaver_train_service as nws
    with patch("subprocess.run", side_effect=nws.subprocess.CalledProcessError(1, ["cmd"])):
        result = nws.run_command(["cmd"], "Test failure")
        assert result is False
