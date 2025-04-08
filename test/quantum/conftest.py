
def assert_static_measurements(sensor, ground_truth):

    assert sensor.last_measurement is None, \
        'Last measurement should be initially None'

    total_steps = sensor.num_steps
    for i in range(total_steps):
        sensor.take_measurement(0, ground_truth)
        sensor.update()

        if i != total_steps - 1:
            assert sensor.last_measurement is None, \
                f'Last measurement should be None at step {i}'

    measurement = sensor.last_measurement
    assert measurement is not None, \
        'Measurement should not be none after final step'