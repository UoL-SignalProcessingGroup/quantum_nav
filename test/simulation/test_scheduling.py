import pytest

from qnav.fusion.altimeter import FixedGainAltimeter
from qnav.fusion.ins import NumericalINS
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.altimeter import Altimeter
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.gyroscope import Gyroscope
from qnav.measurement.sensor import SensorFusion, Sensor
from qnav.simulation.scheduling import TimingHandler


def get_frequencies():
    """
    Returns list of sensor frequencies to cover.
    """
    return 0.1, 1.0, 250.0, 500.0

@pytest.fixture(params=get_frequencies())
def accelerometer(request) -> Accelerometer:
    """
    Perfect fresh accelerometer instance.
    """
    frequency = request.param
    error_profile = ErrorProperties()
    return Accelerometer(frequency, error_profile)

@pytest.fixture(params=get_frequencies())
def gyroscope(request) -> Gyroscope:
    """
    Perfect fresh gyroscope instance.
    """
    frequency = request.param
    error_profile = ErrorProperties()
    return Gyroscope(frequency, error_profile)

@pytest.fixture(params=get_frequencies())
def altimeter(request) -> Altimeter:
    """
    Perfect fresh altimeter instance.
    """
    frequency = request.param
    return Altimeter(frequency)

@pytest.fixture
def ins_fusion(accelerometer, gyroscope) -> NumericalINS:
    """
    INS with accelerometer and gyroscope sensors.
    """
    return NumericalINS(accelerometer, gyroscope)

@pytest.fixture
def altimeter_fusion(altimeter) -> FixedGainAltimeter:
    """
    Fixed gain altimeter fusion.
    """
    return FixedGainAltimeter(altimeter)

@pytest.fixture
def hardware(accelerometer, gyroscope, altimeter):
    """
    Returns test hardware containing INS and altimeter.
    """
    ins = NumericalINS(accelerometer, gyroscope)
    alt = FixedGainAltimeter(altimeter)
    return [ins, alt]

def get_sensor_list(sensor_fusion: list[SensorFusion]):
    """
    Extracts all individual sensor instances from hardware list.
    """
    return [s for sf in sensor_fusion for s in sf.sensors]


def test_next_sensor(hardware):

    # Initialise a timing handler and get output
    th = TimingHandler(hardware)
    next_sensor = th.get_next_sensor()

    # Calculate the expected output options
    sensors = get_sensor_list(hardware)
    min_time = min([s.next_update for s in sensors])
    expected = [s for s in sensors if s.next_update == min_time]

    # Ensure output is in expected list
    assert next_sensor in expected


def test_sensor_used(hardware, random_truth):

    th = TimingHandler(hardware)
    sensors = get_sensor_list(hardware)

    is_used = {
        h: {s: False for s in h.sensors}
        for h in hardware
    }

    # For each sensor in order:
    for _ in range(len(sensors)):

        # Get the next sensor instance:
        sensor = th.get_next_sensor()

        # Ensure the sensor is the next:
        min_time = min([s.next_update for s in sensors])
        assert sensor.next_update == min_time, \
            'Sensor with incorrect next time returned'

        # Update the sensor's measurement and state
        sensor.take_measurement(0, random_truth)
        sensor.update()

        # Mark and record the sensor as used.
        th.set_sensor_used(sensor)

        expected = []
        for h in hardware:

            # If the sensor is used by hardware
            if sensor in is_used[h]:

                # Mark as used
                is_used[h][sensor] = True

                # If all keys are set to true:
                if all(is_used[h].values()):

                    # Reset keys and append hardware to expected
                    is_used[h] = {key: False for key in is_used[h]}
                    expected.append(h)

        # Ensure the ready fusions are as expected
        next_fusion = th.get_next_fusion()
        assert set(next_fusion) == set(expected), \
            'Returned next fusion did not match expected'


def test_next_fusion(hardware, random_truth):

    # Initialise a timing handler and get output
    th = TimingHandler(hardware)
    next_fusion = th.get_next_fusion()

    # At first this should always be empty
    assert len(next_fusion) == 0

    # Identify the sensors to update first:
    sensors = get_sensor_list(hardware)
    min_time = min([s.next_update for s in sensors])
    updated_sensors = [s for s in sensors if s.next_update == min_time]

    # Update these sensors:
    for s in updated_sensors:
        s.take_measurement(0, random_truth)
        th.set_sensor_used(s)
        s.update()

    # Get those whose parents have complete match:
    expected = []
    for sf in hardware:
        if all(s in updated_sensors for s in sf.sensors):
            expected.append(sf)

    # Compare the results
    next_fusion = th.get_next_fusion()
    assert set(next_fusion) == set(expected)

def test_next_fusion_for_sensor(hardware, random_truth):

    # Initialise a timing handler and get output
    th = TimingHandler(hardware)
    sensors = get_sensor_list(hardware)

    # At first this should always be empty
    for s in sensors:
        assert len(th.get_next_fusion(s)) == 0

    # Identify the sensors to update first:
    min_time = min([s.next_update for s in sensors])
    updated_sensors = [s for s in sensors if s.next_update == min_time]

    # Update these sensors:
    for s in updated_sensors:
        s.take_measurement(0, random_truth)
        th.set_sensor_used(s)
        s.update()

    # Calculate the expected results
    expected = set()
    for sf in hardware:
        if all(s in updated_sensors for s in sf.sensors):
            expected.add(sf)

    # For each of the updates sensors,
    # check their fusion is returned:
    for s in updated_sensors:
        results = th.get_next_fusion(s)
        for r in results:
            assert r in expected
            expected.remove(r)


@pytest.mark.parametrize('max_time', (1.0, 10.0, 100.0))
def test_sensor_iterator(hardware, random_truth, max_time: float):

    # Initialise a timing handler and get output
    th = TimingHandler(hardware)
    sensors = get_sensor_list(hardware)

    # For each sensor in the hardware iterator:
    for sensor in th.sensor_iterator(max_time):

        # Check the due time of the expected next sensor
        min_time = min([s.next_update for s in sensors])
        assert min_time <= max_time, \
            'Iteration exceeds maximum time'

        # Check the sensor with the correct time is returned
        assert sensor.next_update == min_time, \
            'Sensor with greater due time than minimum returned'

        # Update and mark the sensor as used.
        sensor.take_measurement(0, random_truth)
        th.set_sensor_used(sensor)
        sensor.update()




