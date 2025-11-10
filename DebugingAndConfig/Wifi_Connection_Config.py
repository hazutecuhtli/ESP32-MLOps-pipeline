import network, time, json
from machine import Pin
import dht
try:
    from umqtt.simple import MQTTClient
except:
    MQTTClient = None  # por si aún no copiaste la lib

SSID = "<Wifi Name>"
PWD  = "<Wifi Password>"
MQTT_HOST = "192.168.1.50"
SENSOR_ID = "esp32-01"

def wifi_connect():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        wlan.connect(SSID, PWD)
        while not wlan.isconnected():
            time.sleep(0.3)
    print("WiFi OK:", wlan.ifconfig())

def run():
    wifi_connect()
    sensor = dht.DHT22(Pin(4))
    client = MQTTClient(SENSOR_ID, MQTT_HOST) if MQTTClient else None

    while True:
        sensor.measure()
        payload = {
            "temp_c": round(sensor.temperature(), 2),
            "hum": round(sensor.humidity(), 2),
            "sensor_id": SENSOR_ID
        }
        print(payload)
        if client:
            try:
                client.connect()
                client.publish(b"home/lab/dht22", json.dumps(payload).encode())
                client.disconnect()
            except Exception as e:
                print("MQTT err:", e)
        time.sleep(10)

run()
