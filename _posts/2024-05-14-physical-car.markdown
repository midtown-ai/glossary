---
layout: post
# layout: single
title:  "Physical car"
date:   2024-05-14 12:51:28 -0800
categories: jekyll update
---

{% include links/all.md %}

* toc
{:toc}


## Links

## Terminology

 Pins - 
  * used to connect shell of car to car
  * used to connect compute layer (top layer) to drive train (bottom layer)
  * <!> Pin loops must be facing away from the car's tires

 Extension - hold the shell above the car, to use as replacement in case you break the one you have (hard crash)
  * 2 types
    * tall
    * small

 Car chassis - assembled car without the shell
  * = compute layer + drive train

 Compute layer - top layer of the DeepRacer car that includes an onboard computer and its compute battery

 Compute Battery - Dell battery to power the compute module
  * Cables / connectors
    * USB cable in USB connector
  * Light indicators
    * 4 blinking LEDs = 
    * All LEDs are on = fully charged
  * Button
    * turn on/off - to power the battery/power bank
  * Other
    * Silicon wrap - to control excess cable you might need for cable between battery and compute car itself.

 Compute module - onboard LINUX computer
  * Cables
    * 4-inch USB-C to USB-C cable to connect the compute battery to the compute module
    * Micro-USB to USB-A cable for point to point connection between car and desktop for initial setup
  * Connectors
    * Right side
      * HDMI video cable - boot process and GUI access
      * Micro-USB (network) - point to point connection between car and desktop for initial setup
      * USB-C (power adapter port) - to power the compute module from a wall transformer or the compute battery
    * Left side
      * micro SD card reader-slot - transfer files to compute module
      * Buttons
      * Status LEDs
  * Light indicators
    * status LED
      * power indicator (front)
        * blank - no power
        * red -
        * flashing green - booting in progress
        * solid green - stuck in booting?
        * blue - successful power on
      * wifi indicator (middle)
        * solid blue - car properly connected to wifi
        * flashing blue - attempting to connect to wifi
        * red - connection to wifi failed
        * blinking blue --> red --> off -
      * ??
    * Tail light - to identify DeepRacer cars when many are around (light can be changed/customized in the web UI)
  * Button
    * power on/off (front)
      * if off - long push turn it on
      * if on - long push turns it off
    * reset indicator (back)

 Compute battery bed - where the compute battery gets velcro'd into!

 Balance charger - Drive battery charger
  * Cables / connectors
    * Has a transformer wall connector
    * White connector used to connect to the Drive battery for charging
  * Light indicators
    * Green - Full charge, ready to go
    * Red light + green light = not fully charged
      * Red - charging?
      * blinking red and blinking green - charging?
      * solid red and blinking green - charging?

 Drive train - bottom layer of the DeepRacer car
  * Connected to top layer using pins when car is properly assembled
  * Components
    * Drive motor (cable and with heat sink)
    * Bed with port (red connector) where drive battery is plugged in
    * steering, suspension, and wheels

 Drive Battery - A lithium battery  pack used to drive the vehicle around the track
  * Cables / connectors
    * White 3-pin connector to charge with (by connecting to balance charger)
    * Red 2-pin connector to run the car with

 Vehicle battery - See Drive battery

 Lockout state - to preserve the battery health, the battery goes into lockout state
  * when this happens, the battery won't power the drive train even if still charged
  * this happens if
    * the drive battery is not turned off after usage
    * the battery power is low and it needs to be recharged
    * the car is not used for a while
  * to prevent this from happening
    * disconnect both cable (red and white) from car
    * fully charge battery
  * to Get out of locked state, use the 'unlock' cable

 Unlock cable - used to unlock drive battery when in a lockout state
  * Cables / connectors
    * White 3-pin JST female connector
    * Red 2-pin JST connector

 Tail light - the compute module is booted up when the tail light lights up

 Ackerman steering - front wheels are not perfectly aligned by design 
  * Used to add velocity and grip in corners!

 Camera port
  * 3 ports available,
  * by default, only one/single front-facing/front-lens camera
  * for stereoscopic vision, use 2 camera at a time
  * <!> You cannot use 3 camera at once because of the physical size of the cameras

 Heat sink 
  * Located in the top layer under compute module, therefore facing down when car is correctly put together


## Docs

### Unboxing

 * unboxing video - [https://aws.amazon.com/deepracer/getting-started/](https://aws.amazon.com/deepracer/getting-started/)

### Assembly

 * car assembly - [https://aws.amazon.com/deepracer/getting-started/](https://aws.amazon.com/deepracer/getting-started/)
   1. charge drive battery
   2. install the drive battery by connect the RED connector to RED input
   3. use velcro hook to secure the drive battery
   4. power on the drive train using the switch on the drive train next to the  front left wheel
      * on position = bring button to left
      * off position = bring button to right
      * 2 beep emitted if drive train receive power
   5. attach the compute layer using the pins (loop away from tire)
   6. connect the compute module to a power adapter (not the laptop battery yet)
   7. power the compute module by (long) pushing the power button
   8. check the status of the power LED in the compute status LED
      * if blank - no power
      * if blinking green - boot sequence in progress
      * if solid green - 
      * if blue - boot sequence completed successfully
   9. 

### Calibration

 * calibrating AWS deepracer - [https://aws.amazon.com/deepracer/getting-started/](https://aws.amazon.com/deepracer/getting-started/)

### Troubleshooting

 * console acces
   * connect desktop to car using USB-A to USB-micro cable
   * turn off wifi on desktop
   * https://hostname.local where hostname is AMSS-1234 (found on sticker under the car)

## More

### outdate doc

 {% pdf "https://d1.awsstatic.com/deepracer/AWS-DeepRacer-Getting-Started-Guide.pdf" %}

 Source - [https://aws.amazon.com/deepracer/getting-started/](https://aws.amazon.com/deepracer/getting-started/)
