"""
test_netweaver_prediction_service.py
Unit tests for netweaver_prediction_service.py
Author: Elham Esmaeilnia
Date: 2025 Oct 29
"""

import pytest
from unittest.mock import patch, MagicMock
import sys

# --------------------- Test 1: Successful prediction flow ---------------------
@patch("subprocess.run")
@patch("sys.exit")
def test_prediction_service_success(mock_exit, mock_run):
    import apps.NetWeaver.src.services.netweaver_prediction_service as nps

    # Mock subprocess.run always succeeds
    mock_run.return_value = MagicMock(returncode=0)

    # Provide fake CLI arguments
    with patch.object(sys, "argv", [
        "netweaver_prediction_service.py",
        "--latest_hours", "6",
        "--future_steps", "5",
        "--save_details",
        "--no_timestamp",
        "--model_path", "path/to/model.pth"
    ]):
        nps.main()

    # sys.exit should not be called on success
    mock_exit.assert_not_called()


# --------------------- Test 2: Pipeline failure triggers sys.exit ---------------------
@patch("subprocess.run")
@patch("sys.exit")
def test_pipeline_failure(mock_exit, mock_run):
    import apps.NetWeaver.src.services.netweaver_prediction_service as nps

    # Fail the data pipeline step
    def mock_run_side_effect(cmd, check, text, capture_output, cwd):
        if any("data_pipeline" in c for c in cmd):
            raise nps.subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return MagicMock(returncode=0)

    mock_run.side_effect = mock_run_side_effect

    with patch.object(sys, "argv", ["netweaver_prediction_service.py", "--latest_hours", "6"]):
        nps.main()
        # Should call sys.exit(1) after pipeline failure
        mock_exit.assert_called_once_with(1)


# --------------------- Test 3: Prediction failure triggers sys.exit ---------------------
@patch("subprocess.run")
@patch("sys.exit")
def test_prediction_failure(mock_exit, mock_run):
    import apps.NetWeaver.src.services.netweaver_prediction_service as nps
    from unittest.mock import MagicMock

    # Fail only the predict step
    def mock_run_side_effect(cmd, check, text, capture_output, cwd):
        if "NetWeaver.src.predict" in cmd:
            raise nps.subprocess.CalledProcessError(returncode=1, cmd=cmd)
        # Data pipeline step succeeds
        return MagicMock(returncode=0)

    mock_run.side_effect = mock_run_side_effect

    with patch.object(sys, "argv", ["netweaver_prediction_service.py", "--latest_hours", "6"]):
        nps.main()
        # Should call sys.exit(1) only after prediction failure
        mock_exit.assert_called_once_with(1)

# --------------------- Test 4: Default arguments ---------------------
@patch("subprocess.run")
@patch("sys.exit")
def test_prediction_default_args(mock_exit, mock_run):
    import apps.NetWeaver.src.services.netweaver_prediction_service as nps

    mock_run.return_value = MagicMock(returncode=0)

    with patch.object(sys, "argv", ["netweaver_prediction_service.py"]):
        nps.main()
        mock_exit.assert_not_called()
