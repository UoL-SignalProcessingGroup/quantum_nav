
from qnav.estimation.state import EstimatedState
from qnav.util.transformations import lla2ned
from qnav.waypoints.trajectory import GroundTruth


def get_error_summary(estimated_state: EstimatedState, ground_truth: GroundTruth) -> dict:

    ned_error = lla2ned(estimated_state.position, ground_truth.position)

    return {
        'position': ned_error,
        'velocity': estimated_state.velocity - ground_truth.velocity,
        'acceleration': estimated_state.acceleration - ground_truth.acceleration,
        'attitude': estimated_state.attitude - ground_truth.attitude,
        'angle_rates': estimated_state.angle_rates - ground_truth.angle_rates
    }
