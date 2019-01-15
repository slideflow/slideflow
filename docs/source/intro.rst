Introduction
============

``simpleble`` is a high-level OO Python package which aims to provide an easy and intuitive way of interacting with nearby Bluetooth Low Energy (BLE) devices (GATT servers). In essence, this package is an extension of the ``bluepy`` package created by Ian Harvey (see `here <https://github.com/IanHarvey/bluepy/>`_)

The aim here was to define a single object which would allow users to perform the various operations performed by the ``bluepy.btle.Peripheral``, ``bluepy.btle.Scanner``, ``bluepy.btle.Service`` and ``bluepy.btle.Characteristic`` classes of ``bluepy``, from one central place. This functionality is facilitated by the ``simpleble.SimpleBleClient`` and ``simpleble.SimpleBleDevice`` classes, where the latter is an extention/subclass of ``bluepy.btle.Peripheral``, combined with properties of ``bluepy.btle.ScanEntry``.

The current implementation has been developed in Python 3 and tested on a Raspberry Pi Zero W, running Raspbian 9 (stretch), but should work with Python 2.7+ (maybe with minor modifications in terms of printing and error handling) and most Debian based OSs.

Motivation
**********

As a newbie experimenter/hobbyist in the field of IoT using BLE communications, I found it pretty hard to identify a Python package which would enable one to use a Raspberry Pi (Zero W inthis case) to swiftly scan, connect to and read/write from/to a nearby BLE device (GATT server).

This package is intended to provide a quick, as well as (hopefully) easy to undestand, way of getting a simple BLE GATT client up and running, for all those out there, who, like myself, are hands-on learners and are eager to get their hands dirty from early on.

Limitations
***********

- As my main use-case scenario was to simply connect two devices, the current version of :class:`simpleble.SimpleBleClient` has been designed and implemented with this use-case in mind. As such, if you are looking for a package to allow you to connect to multiple devices, then know that off-the-self this package DOES NOT allow you to do so. However, implementing such a feature is an easily achievable task, which has been planned for sometime in the near future and if there proves to be interest on the project, I would be happy to speed up the process.

- Only Read and Write operations are currently supported, but I am planning on adding Notifications soon.

- Although the interfacing operations of the :class:`bluepy.btle.Service` and :class:`bluepy.btle.Peripheral` classes have been brought forward to the :class:`simpleble.SimpleBleClient` class, the same has not been done for the :class:`bluepy.btle.Descriptor`, meaning that the :class:`simpleble.SimpleBleClient` cannot be used to directly access the Descriptors. This can however be done easily by obtaining a handle of a :class:`simpleble.SimpleBleDevice` object and calling the superclass :meth:`bluepy.btle.Peripheral.getDescriptors` method.
