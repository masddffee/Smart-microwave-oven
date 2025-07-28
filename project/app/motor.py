#!/usr/bin/python3
import RPi.GPIO as GPIO
import time
import threading

step_sequence = [[1,0,0,1],
                 [1,0,0,0],
                 [1,1,0,0],
                 [0,1,0,0],
                 [0,1,1,0],
                 [0,0,1,0],
                 [0,0,1,1],
                 [0,0,0,1]]
                 
class Motor():
	def __init__(self,in1,in2,in3,in4,step_sleep = 0.001,step_count = 4096*5 ,direction=False):
		GPIO.setmode( GPIO.BCM )
		self.in1 = in1
		self.in2 = in2
		self.in3 = in3
		self.in4 = in4
		GPIO.setup( in1, GPIO.OUT )
		GPIO.setup( in2, GPIO.OUT )
		GPIO.setup( in3, GPIO.OUT )
		GPIO.setup( in4, GPIO.OUT )
		GPIO.output( in1, GPIO.LOW )
		GPIO.output( in2, GPIO.LOW )
		GPIO.output( in3, GPIO.LOW )
		GPIO.output( in4, GPIO.LOW )
		self.motor_pins = [in1,in2,in3,in4]
		self.motor_step_counter = 0 
		self.running = False
		self.fb = True
		self.step_count = step_count
		self.step_sleep = step_sleep
		
		
	def forward(self):
		self.fb = True
		if not self.running:
			self.start()
	
	def backward(self):
		self.fb = False
		if not self.running:
			self.start()
	
	def start(self):
		self.running = True
		self.startjob()
		
	def startjob(self):
		motorTherad = threading.Thread(target=self.mainloop)
		motorTherad.start()
		
	def mainloop(self):
		while self.running:
			i = 0
			for i in range(self.step_count):
				for pin in range(0, len(self.motor_pins)):
					GPIO.output( self.motor_pins[pin], step_sequence[self.motor_step_counter][pin] )
				if self.fb:
					self.motor_step_counter = (self.motor_step_counter - 1) % 8
				elif not self.fb:
					self.motor_step_counter = (self.motor_step_counter + 1) % 8
				time.sleep( self.step_sleep )
				if not self.running :
					break
				
		
		
	def stop(self):
		self.running = False
		GPIO.output( self.in1, GPIO.LOW )
		GPIO.output( self.in2, GPIO.LOW )
		GPIO.output( self.in3, GPIO.LOW )
		GPIO.output( self.in4, GPIO.LOW )


if __name__ == "__main__":
	in1 = 19  # 腳位
	in2 = 13
	in3 = 6
	in4 = 5
	motor1 = Motor(in1,in2,in3,in4)
	motor1.forward()
	time.sleep(3)
	motor1.stop()
	
