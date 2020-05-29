### Advertisement insertion application


The mechanism for replacing advertising uses the methods of computer vision and artificial intelligence.


In order to perform the replacement, you must have the following:
 - The image to be inserted
 - Video that will be processed
 - Installed application
 - Nvidia drivers

This setting is performed only at the first start.
To run the application, you need to do the following:
 - Download the ```build.sh``` file from the **source** to the folder where the application should be located
 - Execute the command ```bash build.sh``` in terminal (**This setting is performed only at the first start**)
 - Keep track of program requests, as verification of access rights may be required.


**You can now run the program by running the command: ```bash run.sh``` from the program folder.**

**Each subsequent application is launched by executing a command: ```bash run.sh``` from the application's folder.**

After executing this command, you started the server, now you need to go to the [link](http://0.0.0.0:5089/)

To **select a video** for processing you need move source video in folder ```instance/upload/``` and write video's name.

The next step is to select the time periods to replace the banner. Data is transmitted as start and end periods, note that **time intervals are transmitted in seconds**. For example, 2 minutes 10 seconds in a query will correspond to a value of 130 seconds.


Then you can choose logo for replacement. Upload your own logo according to the brand you want to change:
 - gazprom
 - heineken
 - mastercard
 - nissan
 - pepsi
 - playstation

After completing all these requests, you need to send a command to start processing.
