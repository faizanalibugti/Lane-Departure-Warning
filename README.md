# Lane Departure Warning System

1. Download or clone this repository
2. Navigate to the repo on your local disk using Anaconda Prompt (cd)
3. **pip install playsound** 
4. Run **python LDW.py**

# Tips:

- It is possible that isGreen() may require tuning of its thresholds
- If the warning is too annoying comment out **lines 62-63** of LDW.py
- Modify **line 123** to change the screen capture region
    By default, it is: screen = grab_screen(region=(0, 40, 1000, 600))

Screen capture frame is first enhanced using adaptive gamma correction

The enhanced image is passed to ego lane segmentation model

The green masked image that is the output of the ego lane segmentation is passed to
the inGreen() function which determines whether LDW sound be triggered or not

The green masked image is superimposed on the original image
