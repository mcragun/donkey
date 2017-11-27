import time
from Adafruit_BNO055 import BNO055

class Bno055:
    '''
    Using Adafruit BNO055 imu
    for instruction on setup:
    https://learn.adafruit.com/bno055-absolute-orientation-sensor-with-raspberry-pi-and-beaglebone-black/software
    
    install adafruit lib:
    git clone https://github.com/adafruit/Adafruit_Python_BNO055.git
    pip install Adafruit_Python_BNO055.git
    pip install RPi.GPIO
    '''

    def __init__(self, poll_delay=0.0166):
        from Adafruit_BNO055 import BNO055
        self.sensor = BNO055.BNO055(rst=12)
        self.accel = { 'x' : 0., 'y' : 0., 'z' : 0. }
        self.gyro = { 'x' : 0., 'y' : 0., 'z' : 0. }
        self.temp = 0.
        self.heading = 0.
        self.roll = 0.
        self.pitch = 0.
        self.poll_delay = poll_delay
        self.on = True

    def update(self):
        while self.on:
            self.poll()
            time.sleep(self.poll_delay)
                
    def poll(self):
        self.accel_raw = self.sensor.read_accelerometer()
        self.accel['x'] = self.accel_raw[0]
        self.accel['y'] = self.accel_raw[1]
        self.accel['z'] = self.accel_raw[2]

        self.gyro_raw = self.sensor.read_gyroscope()
        self.gyro['x'] = self.gyro_raw[0]
        self.gyro['y'] = self.gyro_raw[1]
        self.gyro['z'] = self.gyro_raw[2]

        self.temp = self.sensor.read_temp()      

        self.heading, self.roll, self.pitch = self.sensor.read_euler()

    def run_threaded(self):
        return self.accel['x'], self.accel['y'], self.accel['z'], self.gyro['x'], self.gyro['y'], self.gyro['z'], self.temp

    def run(self):
        self.poll()
        return self.accel['x'], self.accel['y'], self.accel['z'], self.gyro['x'], self.gyro['y'], self.gyro['z'], self.temp

    def shutdown(self):
        self.on = False


if __name__ == "__main__":
    iter = 0
    p = Bno055()
    while iter < 100:
        data = p.run()
        print(data)
        time.sleep(0.1)
        iter += 1
     
