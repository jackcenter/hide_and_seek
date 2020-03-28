# Hide and Seek
Hide and Seek is a project based on the paper "Distributed Data Fusion" by Mark Campbell and Nisar Ahmed. The purpose of
this project is to show how information can be shared among a network of agents to improve the state estimate of all 
agents on the network. To explore this, a simple example based on a game of hide and seek that continues to get more 
elaborate. Currently, the following examples are available:
1. Static Process, Stationary Linear Sensors

## How to Install
The project is written in python3 and the following instructions are for Windows. For MAC or Linux, use the operating 
specific language in place of Windows commands. To install, 
[download the ZIP](https://github.com/jackcenter/hide_and_seek/archive/master.zip) from github and unpack it on your 
computer.

This project is dependent on Matplotlib, NumPy, and SciPy. To install the required dependencies, open a terminal, 
navigate to the directory where the project was saved, and install requirements.txt by typing the following:

>$ pip install -r requirements.txt

## Example
The project is currently run from the command line. To implement and example, open a terminal, navigate to the location
where the project is saved, and enter the following:

>$ python program.py

Then follow the prompts to view the desired example. The following is a snapshot of the animation of:

>$ This program runs the following exercises: \
>$ &nbsp; [1]: Static Process, Stationary Linear Sensors \
>$ \
>$ Select an exercise would you like to run: 1 

![Example image](/images/Example.png) 

The seekers' positions on the left side of the map are marked by x's. The hider's position is marked by a red x on the
right side of the screen. The seekers are receiving noisy, linear 2-D position measurements from the hider, which is not
particularly realistic since most hiders wouldn't want to knowingly broadcast their position to those looking for them.
However, this example serves as a straight forward starting point to implement the channel filter. The noise on the 
sensors was modeled as independent Additive White Gaussian Noise on both the position measurements for x and y. Seeker 1
has a covariance defined by diag([250 m^2, 250 m^2]) while the remaining seekers have covariances defined by
diag([500 m^2, 500 m^2]). This means that Seeker 1 has a significantly more reliable sensor compared to the 
rest of the seekers, though none of the sensors are particularly good. 

For this example, Seeker 1, pictured at the top left of the map at (-40, 40), is estimating the state of the hider at 
(30, 0) independently from the other seekers. The Control Position located at (-25, 5) is a Pseudo-Seeker that is meant 
to directly compare the results from Seeker 3 at (-40, 0). This Pseudo-Seeker uses the exact same measurement generated 
by Seeker 3, but does not have the benefit of sharing information with the network. The remaining seekers, Seeker 2 
through Seeker 5, share information according to the dashed blue lined that connects them (ie. Seeker 2 (-20, 20) can 
only communicate with Seeker 3, Seeker 5 (-40, -40) can only communicate with Seeker 4 (-20, -5), whereas Seeker 3 
communicates with Seeker 2 and Seeker 4).

The color coordinated diamonds on the right side of the screen represent the 2-D state estimate from the seeker with the 
corresponding color. Additionally, the ellipse with the same color represents the associated 2-D, two sigma (95%) error
bound. This means the seeker is 95% confident the true position of the hider is within this ellipse. Individual 
measurements are plotted as small squares for each time step. In the example figure, the trends are as expected, with 
the networked seekers' estimates being extremely close to the true position with error bounds much smaller than those of
the independent seekers (Seeker 1 and the Pseudo-Seeker). Additionally, the independent seeker outperforms the 
Pseudo-Seeker as expected due to the independent seeker having a significantly better sensor.

## Reporting Issues
Use the [GitHub issue tracker](https://github.com/jackcenter/hide_and_seek/issues) for:
* Bug reports
* Documentation issues
* Feature requests

## License
This project operates under the MIT License. For more information see 
[LICENSE](https://github.com/jackcenter/hide_and_seek/blob/master/LICENSE.txt).

## Author
Jack Center
