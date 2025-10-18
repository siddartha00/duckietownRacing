# Template: template-ros

This template provides a boilerplate repository
for developing ROS-based software in Duckietown.

**NOTE:** If you want to develop software that does not use
ROS, check out [this template](https://github.com/duckietown/template-basic).


## How to use it

### 1. Fork this repository

Use the fork button in the top-right corner of the github page to fork this template repository.


### 2. Create a new repository

Create a new repository on github.com while
specifying the newly forked template repository as
a template for your new repository.


### 3. Define dependencies

List the dependencies in the files `dependencies-apt.txt` and
`dependencies-py3.txt` (apt packages and pip packages respectively).


### 4. Place your code

Place your code in the directory `/packages/` of
your new repository.


### 5. Setup launchers

The directory `/launchers` can contain as many launchers (launching scripts)
as you want. A default launcher called `default.sh` must always be present.

If you create an executable script (i.e., a file with a valid shebang statement)
a launcher will be created for it. For example, the script file 
`/launchers/my-launcher.sh` will be available inside the Docker image as the binary
`dt-launcher-my-launcher`.

When launching a new container, you can simply provide `dt-launcher-my-launcher` as
command.



# Basic Insturctions for Accessing development Environment

To access `raspberryPi` from VSCode, use hostname as `siddartha@PiBox.local` and password as `siddartha`
The repository is present in `duckietownRacing`.

# Basic Instructions to build and run docker container on duckiebot using `dts`

## Container Tools in VSCode

1) Install `Container Tools` extension in VSCode.
2) Open Containers tab by clicking on containers button.
3) Extend the `Docker contexts` and right click on `duckeibot12`.
4) Select use from the dropdown menu.

## Building the Docker container image after adding code

To build the docker image locally on `raspberryPi` use

    dts devel build -f

To build the docker image on `duckiebot12` use

    dts devel build -H duckiebot12 -f

## Running the Docker container image after adding code

To run the docker image on `duckiebot12`

    dts devel run -H duckiebot12 -L launch
