from gpiozero import Motor
from time import sleep

mouth = Motor(forward=5, backward=6, enable=12)
body = Motor(forward=19, backward=26, enable=13)

print("mouth one way")
mouth.forward(0.6)
sleep(0.3)
mouth.stop()
sleep(1)

print("mouth other way")
mouth.backward(0.6)
sleep(0.3)
mouth.stop()
sleep(1)

print("body one way")
body.forward(0.6)
sleep(0.4)
body.stop()
sleep(1)

print("body other way")
body.backward(0.6)
sleep(0.4)
body.stop()
