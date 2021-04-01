
#include <webots/Robot.hpp>
#include <webots/DistanceSensor.hpp>
#include <webots/Motor.hpp>

# define TIME_STEP 64
using namespace webots;

int main(int argc, char **argv){
    Robot *robot = new Robot();

    while (robot->step(TIME_STEP)!=-1){
        
    }
    delete robot;
    return 0;
}