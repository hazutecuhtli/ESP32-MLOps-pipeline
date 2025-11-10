from machine import Pin
import dht, time

sensor = dht.DHT22(Pin(4))  # cambia 4 si usaste otro pin
while True:
    sensor.measure()
    temp = sensor.temperature()
    hum = sensor.humidity()
    print("🌡️ Temp:", temp, "°C   💧 Hum:", hum, "%")
    time.sleep(2)
