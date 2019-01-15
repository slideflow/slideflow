Examples
=============

Installation/Usage:
*******************
As the package has not been published on PyPi yet, it CANNOT be install using pip.

For now, the suggested method is to put the file `simpleble.py` in the same directory as your source files and call ``from simpleble import SimpleBleClient, SimpleBleDevice``.

``bluepy`` must also be installed and imported as shown in the example below.
For instructions about how to install, as well as the full documentation of, ``bluepy`` please refer `here <https://github.com/IanHarvey/bluepy/>`_

Search for device, connect and read characteristic
**************************************************
.. code-block:: python

    """This example demonstrates a simple BLE client that scans for devices,
    connects to a device (GATT server) of choice and continuously reads a characteristic on that device.

    The GATT Server in this example runs on an ESP32 with Arduino. For the
    exact script used for this example see `here <https://github.com/nkolban/ESP32_BLE_Arduino/blob/6bad7b42a96f0aa493323ef4821a8efb0e8815f2/examples/BLE_notify/BLE_notify.ino/>`_
    """

    from bluepy.btle import *
    from simpleble import SimpleBleClient, SimpleBleDevice

    # The UUID of the characteristic we want to read and the name of the device # we want to read it from
    Characteristic_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
    Device_Name = "MyESP32"

    # Define our scan and notification callback methods
    def myScanCallback(client, device, isNewDevice, isNewData):
        client._yes = True
        print("#MAC: " + device.addr + " #isNewDevice: " +
            str(isNewDevice) + " #isNewData: " + str(isNewData))
    # TODO: NOTIFICATIONS ARE NOT SUPPORTED YET
    # def myNotificationCallback(client, characteristic, data):
    #     print("Notification received!")
    #     print("  Characteristic UUID: " + characteristic.uuid)
    #     print("  Data: " + str(data))

    # Instantiate a SimpleBleClient and set it's scan callback
    bleClient = SimpleBleClient()
    bleClient.setScanCallback(myScanCallback)
    # TODO: NOTIFICATIONS ARE NOT SUPPORTED YET
    # bleClient.setNotificationCallback(myNotificationCollback)

    # Error handling to detect Keyboard interrupt (Ctrl+C)
    # Loop to ensure we can survive connection drops
    while(not bleClient.isConnected()):
        try:
            # Search for 2 seconds and return a device of interest if found.
            # Internally this makes a call to bleClient.scan(timeout), thus
            # triggering the scan callback method when nearby devices are detected
            device = bleClient.searchDevice(name="MyESP32", timeout=2)
            if(device is not None):
                # If the device was found print out it's info
                print("Found device!!")
                device.printInfo()

                # Proceed to connect to the device
                print("Proceeding to connect....")
                if(bleClient.connect(device)):

                    # Have a peek at the services provided by the device
                    services = device.getServices()
                    for service in services:
                        print("Service ["+str(service.uuid)+"]")

                    # Check to see if the device provides a characteristic with the
                    # desired UUID
                    counter = bleClient.getCharacteristics(
                        uuids=[Characteristic_UUID])[0]
                    if(counter):
                        # If it does, then we proceed to read its value every second
                        while(True):
                            # Error handling ensures that we can survive from
                            # potential connection drops
                            try:
                                # Read the data as bytes and convert to string
                                data_bytes = bleClient.readCharacteristic(
                                    counter)
                                data_str = "".join(map(chr, data_bytes))

                                # Now print the data and wait for a second
                                print("Data: " + data_str)
                                time.sleep(1.0)
                            except BTLEException as e:
                                # If we get disconnected from the device, keep
                                # looping until we have reconnected
                                if(e.code == BTLEException.DISCONNECTED):
                                    bleClient.disconnect()
                                    print(
                                        "Connection to BLE device has been lost!")
                                    break
                                    # while(not bleClient.isConnected()):
                                    #     bleClient.connect(device)

                else:
                    print("Could not connect to device! Retrying in 3 sec...")
                    time.sleep(3.0)
            else:
                print("Device not found! Retrying in 3 sec...")
                time.sleep(3.0)
        except BTLEException as e:
            # If we get disconnected from the device, keep
            # looping until we have reconnected
            if(e.code == BTLEException.DISCONNECTED):
                bleClient.disconnect()
                print(
                    "Connection to BLE device has been lost!")
                break
        except KeyboardInterrupt as e:
            # Detect keyboard interrupt and close down
            # bleClient gracefully
            bleClient.disconnect()
            raise e
